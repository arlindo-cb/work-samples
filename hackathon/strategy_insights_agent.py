# ============================================================
# STRATEGIC INSIGHTS AGENT FOR EXECUTIVE DECISION-MAKING
# ============================================================

from pyspark.sql import DataFrame, functions as F, Window
from pyspark.sql.types import *
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class StrategyInsightsAgent:
    """
    Analyzes optimized dispatch results to generate strategic insights
    for executive decision-making.
    
    Provides:
    - Geographic skill shortage analysis
    - Time-of-day travel pattern analysis
    - Technician utilization patterns
    - Actionable recommendations for resource allocation
    """
    
    def __init__(
        self,
        spark,
        catalog: str = "hackathon",
        schema: str = "hackathon_fiber_vault"
    ):
        """Initialize the strategy insights agent."""
        self.spark = spark
        self.catalog = catalog
        self.schema = schema
        self.table_prefix = f"{catalog}.{schema}"
        
        print("=" * 80)
        print("STRATEGIC INSIGHTS AGENT")
        print("=" * 80)
        print(f"Catalog: {catalog}")
        print(f"Schema: {schema}")
        print("=" * 80)
    
    # ============================================================
    # DATA LOADING
    # ============================================================
    
    def load_data(self) -> Tuple[DataFrame, DataFrame]:
        """Load optimized dispatches and technician data."""
        print("\n[1/7] Loading data...")
        
        self.dispatches_df = self.spark.table(f"{self.table_prefix}.current_dispatches_hackathon_opt")
        self.technicians_df = self.spark.table(f"{self.table_prefix}.technicians_hackathon")
        
        dispatch_count = self.dispatches_df.count()
        tech_count = self.technicians_df.count()
        
        print(f"  ✓ Optimized Dispatches: {dispatch_count:,}")
        print(f"  ✓ Technicians: {tech_count:,}")
        
        return self.dispatches_df, self.technicians_df
    
    # ============================================================
    # ANALYSIS 1: GEOGRAPHIC SKILL SHORTAGE ANALYSIS
    # ============================================================
    
    def analyze_skill_shortages_by_city(self) -> pd.DataFrame:
        """
        Identify cities with skill shortages based on:
        - Unassigned dispatches
        - Average travel distance
        - Fallback assignments (constraints not met)
        """
        print("\n[2/7] Analyzing geographic skill shortages...")
        
        # Aggregate by city and skill
        city_skill_analysis = self.dispatches_df.groupBy(
            "City", "Required_skill"
        ).agg(
            F.count("*").alias("total_dispatches"),
            F.sum(F.when(F.col("Optimized_technician_id").isNull(), 1).otherwise(0)).alias("unassigned"),
            F.sum(F.when(F.col("All_constraints_met") == False, 1).otherwise(0)).alias("fallback_assignments"),
            F.avg("Optimized_distance_km").alias("avg_distance_km"),
            F.max("Optimized_distance_km").alias("max_distance_km"),
            F.avg("Optimized_utilization").alias("avg_tech_utilization")
        )
        
        city_skill_analysis = city_skill_analysis.withColumn(
            "unassigned_rate",
            F.col("unassigned") / F.col("total_dispatches")
        ).withColumn(
            "fallback_rate",
            F.col("fallback_assignments") / F.col("total_dispatches")
        ).withColumn(
            "problem_score",
            (F.col("unassigned_rate") * 3 + F.col("fallback_rate") * 2 + 
             F.when(F.col("avg_distance_km") > 30, 1).otherwise(0))
        )
        
        city_skill_pd = city_skill_analysis.toPandas()
        
        # Identify top problem areas
        top_problems = city_skill_pd.nlargest(10, 'problem_score')
        
        print(f"  ✓ Analyzed {len(city_skill_pd)} city-skill combinations")
        print(f"  ✓ Identified {len(top_problems)} critical shortage areas")
        
        return city_skill_pd
    
    # ============================================================
    # ANALYSIS 2: TIME-OF-DAY TRAVEL PATTERN ANALYSIS
    # ============================================================
    
    def analyze_time_of_day_patterns(self) -> pd.DataFrame:
        """
        Analyze travel distance and assignment success by time of day.
        """
        print("\n[3/7] Analyzing time-of-day patterns...")
        
        # Extract hour from appointment start time
        time_analysis = self.dispatches_df.withColumn(
            "appointment_hour",
            F.hour(F.col("Appointment_start_datetime"))
        ).withColumn(
            "time_window",
            F.when(F.col("appointment_hour") < 8, "Early (6-8 AM)")
            .when(F.col("appointment_hour") < 12, "Morning (8-12 PM)")
            .when(F.col("appointment_hour") < 16, "Afternoon (12-4 PM)")
            .when(F.col("appointment_hour") < 20, "Evening (4-8 PM)")
            .otherwise("Late (8+ PM)")
        )
        
        time_patterns = time_analysis.groupBy("time_window", "appointment_hour").agg(
            F.count("*").alias("total_dispatches"),
            F.avg("Optimized_distance_km").alias("avg_distance_km"),
            F.percentile_approx("Optimized_distance_km", 0.90).alias("p90_distance_km"),
            F.sum(F.when(F.col("All_constraints_met") == False, 1).otherwise(0)).alias("fallback_count"),
            F.avg("Optimized_p_productive").alias("avg_productivity"),
            F.avg("Optimized_utilization").alias("avg_utilization")
        ).orderBy("appointment_hour")
        
        time_patterns_pd = time_patterns.toPandas()
        
        print(f"  ✓ Analyzed patterns across {len(time_patterns_pd)} time windows")
        
        return time_patterns_pd
    
    # ============================================================
    # ANALYSIS 3: TECHNICIAN UTILIZATION ANALYSIS
    # ============================================================
    
    def analyze_technician_utilization(self) -> pd.DataFrame:
        """
        Identify over-utilized and under-utilized technicians.
        """
        print("\n[4/7] Analyzing technician utilization...")
        
        # Get assignments per technician
        tech_assignments = self.dispatches_df.filter(
            F.col("Optimized_technician_id").isNotNull()
        ).groupBy("Optimized_technician_id").agg(
            F.count("*").alias("assignments"),
            F.avg("Optimized_distance_km").alias("avg_distance"),
            F.sum("Duration_min").alias("total_duration_min"),
            F.avg("Optimized_p_productive").alias("avg_productivity"),
            F.first("Optimized_utilization").alias("utilization"),
            F.first("City").alias("base_city")
        )
        
        # Join with technician details
        tech_analysis = tech_assignments.join(
            self.technicians_df,
            tech_assignments.Optimized_technician_id == self.technicians_df.Technician_id,
            "left"
        ).select(
            "Optimized_technician_id",
            F.col("Name").alias("technician_name"),
            F.col("Primary_skill").alias("skill"),
            F.col("City").alias("city"),
            "assignments",
            F.col("Workload_capacity").alias("capacity"),
            "utilization",
            "avg_distance",
            "total_duration_min",
            "avg_productivity"
        )
        
        tech_analysis = tech_analysis.withColumn(
            "utilization_category",
            F.when(F.col("utilization") < 0.4, "Under-utilized (<40%)")
            .when(F.col("utilization") < 0.7, "Well-balanced (40-70%)")
            .when(F.col("utilization") < 0.9, "High (70-90%)")
            .otherwise("Over-utilized (>90%)")
        )
        
        tech_analysis_pd = tech_analysis.toPandas()
        
        print(f"  ✓ Analyzed {len(tech_analysis_pd)} active technicians")
        
        return tech_analysis_pd
    
    # ============================================================
    # ANALYSIS 4: CONSTRAINT FAILURE PATTERNS
    # ============================================================
    
    def analyze_constraint_failures(self) -> pd.DataFrame:
        """
        Analyze which constraints fail most frequently and where.
        """
        print("\n[5/7] Analyzing constraint failure patterns...")
        
        constraint_analysis = self.dispatches_df.filter(
            F.col("All_constraints_met") == False
        ).groupBy("City", "Required_skill", "Optimization_constraint_status").agg(
            F.count("*").alias("failure_count")
        ).orderBy(F.desc("failure_count"))
        
        constraint_pd = constraint_analysis.toPandas()
        
        print(f"  ✓ Analyzed {len(constraint_pd)} constraint failure patterns")
        
        return constraint_pd
    
    # ============================================================
    # ANALYSIS 5: COST OPPORTUNITY ANALYSIS
    # ============================================================
    
    def analyze_cost_opportunities(self) -> Dict:
        """
        Calculate potential savings from optimization improvements.
        """
        print("\n[6/7] Analyzing cost optimization opportunities...")
        
        # Current state metrics
        current_metrics = self.dispatches_df.agg(
            F.avg("Optimized_travel_cost").alias("avg_travel_cost"),
            F.avg("Optimized_distance_km").alias("avg_distance"),
            F.sum(F.when(F.col("All_constraints_met") == False, 1).otherwise(0)).alias("fallback_count"),
            F.count("*").alias("total_dispatches")
        ).collect()[0]
        
        # Calculate improvement potential
        high_distance_dispatches = self.dispatches_df.filter(
            F.col("Optimized_distance_km") > 40
        ).count()
        
        fallback_dispatches = self.dispatches_df.filter(
            F.col("All_constraints_met") == False
        ).count()
        
        opportunities = {
            'current_avg_distance_km': float(current_metrics['avg_distance']),
            'current_avg_travel_cost': float(current_metrics['avg_travel_cost']),
            'high_distance_count': high_distance_dispatches,
            'fallback_count': fallback_dispatches,
            'total_dispatches': int(current_metrics['total_dispatches']),
            'improvement_potential_pct': (high_distance_dispatches / current_metrics['total_dispatches']) * 100
        }
        
        print(f"  ✓ Identified improvement opportunities")
        
        return opportunities
    
    # ============================================================
    # GENERATE STRATEGIC MEMO
    # ============================================================
    
    def generate_strategy_memo(self) -> str:
        """
        Generate executive strategy memo with actionable insights.
        """
        print("\n[7/7] Generating strategic insights memo...")
        
        # Run all analyses
        skill_shortages = self.analyze_skill_shortages_by_city()
        time_patterns = self.analyze_time_of_day_patterns()
        tech_utilization = self.analyze_technician_utilization()
        constraint_failures = self.analyze_constraint_failures()
        cost_opportunities = self.analyze_cost_opportunities()
        
        # Build memo
        memo_lines = []
        
        # Header
        memo_lines.append("=" * 80)
        memo_lines.append("STRATEGIC DISPATCH OPTIMIZATION MEMO")
        memo_lines.append("Executive Summary & Recommendations")
        memo_lines.append("=" * 80)
        memo_lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        memo_lines.append(f"Analysis Period: Current dispatch backlog ({cost_opportunities['total_dispatches']} dispatches)")
        memo_lines.append("=" * 80)
        
        # Section 1: Geographic Skill Shortages
        memo_lines.append("\n📍 SECTION 1: GEOGRAPHIC SKILL SHORTAGE ANALYSIS")
        memo_lines.append("-" * 80)
        
        top_shortages = skill_shortages.nlargest(5, 'problem_score')
        
        if len(top_shortages) > 0:
            memo_lines.append("\n🚨 CRITICAL SHORTAGE AREAS:")
            for idx, row in top_shortages.iterrows():
                if row['problem_score'] > 1:
                    unassigned_pct = row['unassigned_rate'] * 100
                    fallback_pct = row['fallback_rate'] * 100
                    avg_dist = row['avg_distance_km']
                    
                    memo_lines.append(f"\n  • City: {row['City']} | Skill: {row['Required_skill']}")
                    memo_lines.append(f"    - {row['total_dispatches']} dispatches | {unassigned_pct:.0f}% unassigned | {fallback_pct:.0f}% fallback")
                    memo_lines.append(f"    - Average travel: {avg_dist:.1f} km")
                    
                    # Generate recommendation
                    if unassigned_pct > 20:
                        memo_lines.append(f"    ✅ RECOMMENDATION: Add 2-3 '{row['Required_skill']}' technicians in {row['City']}")
                        memo_lines.append(f"       Expected impact: Reduce unassigned rate from {unassigned_pct:.0f}% to <10%")
                    elif avg_dist > 35:
                        memo_lines.append(f"    ✅ RECOMMENDATION: Relocate 1 '{row['Required_skill']}' tech to {row['City']}")
                        memo_lines.append(f"       Expected impact: Reduce avg travel from {avg_dist:.1f} km to ~20 km (save ~{(avg_dist-20)*0.5:.0f} min/dispatch)")
                    elif fallback_pct > 30:
                        memo_lines.append(f"    ✅ RECOMMENDATION: Increase capacity or shift schedules for '{row['Required_skill']}' in {row['City']}")
                        memo_lines.append(f"       Expected impact: Improve constraint satisfaction from {100-fallback_pct:.0f}% to >85%")
        else:
            memo_lines.append("\n✓ No critical geographic skill shortages identified")
        
        # Section 2: Time-of-Day Patterns
        memo_lines.append("\n\n⏰ SECTION 2: TIME-OF-DAY TRAVEL PATTERN ANALYSIS")
        memo_lines.append("-" * 80)
        
        # Find peak travel times
        peak_times = time_patterns.nlargest(3, 'avg_distance_km')
        
        memo_lines.append("\n🚨 PEAK TRAVEL DISTANCE WINDOWS:")
        for idx, row in peak_times.iterrows():
            memo_lines.append(f"\n  • {row['time_window']} ({row['appointment_hour']}:00)")
            memo_lines.append(f"    - {row['total_dispatches']} dispatches | Avg distance: {row['avg_distance_km']:.1f} km")
            memo_lines.append(f"    - P90 distance: {row['p90_distance_km']:.1f} km | {row['fallback_count']} fallback assignments")
            
            if row['avg_distance_km'] > 30:
                time_window = row['time_window']
                improvement = (row['avg_distance_km'] - 20) / row['avg_distance_km'] * 100
                memo_lines.append(f"    ✅ RECOMMENDATION: Add {int(row['fallback_count'] / 20) + 1} technicians during {time_window}")
                memo_lines.append(f"       Expected impact: Reduce travel distance by ~{improvement:.0f}% (save {improvement*0.5:.0f} min avg/dispatch)")
        
        # Section 3: Technician Utilization
        memo_lines.append("\n\n👥 SECTION 3: TECHNICIAN WORKFORCE UTILIZATION")
        memo_lines.append("-" * 80)
        
        # Over-utilized
        over_utilized = tech_utilization[tech_utilization['utilization'] > 0.9]
        under_utilized = tech_utilization[tech_utilization['utilization'] < 0.4]
        
        memo_lines.append(f"\n📊 UTILIZATION SUMMARY:")
        memo_lines.append(f"  • Total active technicians: {len(tech_utilization)}")
        memo_lines.append(f"  • Over-utilized (>90%): {len(over_utilized)} technicians")
        memo_lines.append(f"  • Under-utilized (<40%): {len(under_utilized)} technicians")
        memo_lines.append(f"  • Well-balanced (40-90%): {len(tech_utilization) - len(over_utilized) - len(under_utilized)} technicians")
        
        if len(over_utilized) > 0:
            memo_lines.append(f"\n🚨 OVER-UTILIZED TECHNICIANS (Risk of burnout):")
            for idx, row in over_utilized.head(5).iterrows():
                memo_lines.append(f"  • {row['technician_name']} ({row['Optimized_technician_id']}) - {row['city']}")
                memo_lines.append(f"    - Utilization: {row['utilization']*100:.0f}% | {row['assignments']} assignments | Skill: {row['skill']}")
                memo_lines.append(f"    ✅ RECOMMENDATION: Redistribute {int(row['assignments']*0.2)} assignments to under-utilized techs")
        
        if len(under_utilized) > 0:
            memo_lines.append(f"\n💡 UNDER-UTILIZED TECHNICIANS (Capacity opportunity):")
            total_spare_capacity = under_utilized['capacity'].sum() - under_utilized['assignments'].sum()
            memo_lines.append(f"  • {len(under_utilized)} technicians with spare capacity")
            memo_lines.append(f"  • Total spare capacity: ~{total_spare_capacity:.0f} additional dispatches possible")
            memo_lines.append(f"  ✅ RECOMMENDATION: Cross-train or relocate to shortage areas identified in Section 1")
        
        # Section 4: Cost Optimization
        memo_lines.append("\n\n💰 SECTION 4: COST OPTIMIZATION OPPORTUNITIES")
        memo_lines.append("-" * 80)
        
        memo_lines.append(f"\n📊 CURRENT STATE:")
        memo_lines.append(f"  • Average travel distance: {cost_opportunities['current_avg_distance_km']:.1f} km")
        memo_lines.append(f"  • High-distance dispatches (>40 km): {cost_opportunities['high_distance_count']} ({cost_opportunities['improvement_potential_pct']:.1f}%)")
        memo_lines.append(f"  • Fallback assignments: {cost_opportunities['fallback_count']} ({cost_opportunities['fallback_count']/cost_opportunities['total_dispatches']*100:.1f}%)")
        
        # Calculate potential savings
        potential_distance_savings = cost_opportunities['high_distance_count'] * (cost_opportunities['current_avg_distance_km'] - 25) * 0.5  # minutes saved
        potential_cost_savings = potential_distance_savings * 0.75  # $0.75/min labor cost
        
        memo_lines.append(f"\n💡 OPTIMIZATION POTENTIAL:")
        memo_lines.append(f"  • Potential time savings: ~{potential_distance_savings/60:.0f} hours/week")
        memo_lines.append(f"  • Estimated cost reduction: ~${potential_cost_savings:.0f}/week")
        memo_lines.append(f"  • Fallback improvement potential: {cost_opportunities['fallback_count']} dispatches could meet all constraints")
        
        # Section 5: Top Priority Recommendations
        memo_lines.append("\n\n🎯 SECTION 5: TOP PRIORITY EXECUTIVE ACTIONS")
        memo_lines.append("-" * 80)
        
        memo_lines.append("\n1. IMMEDIATE (Week 1-2):")
        
        # Recommendation 1: Worst skill shortage
        if len(top_shortages) > 0:
            worst = top_shortages.iloc[0]
            memo_lines.append(f"   • Add 2-3 '{worst['Required_skill']}' technicians in {worst['City']}")
            memo_lines.append(f"     Impact: Address {worst['total_dispatches']} dispatches, reduce unassigned from {worst['unassigned_rate']*100:.0f}% to <10%")
        
        # Recommendation 2: Redistribution
        if len(over_utilized) > 0 and len(under_utilized) > 0:
            memo_lines.append(f"   • Redistribute ~{int(over_utilized['assignments'].sum() * 0.15)} assignments from over-utilized to under-utilized techs")
            memo_lines.append(f"     Impact: Balance workload, prevent burnout, utilize spare capacity")
        
        memo_lines.append("\n2. SHORT-TERM (Month 1-2):")
        
        # Peak time staffing
        if len(peak_times) > 0:
            peak = peak_times.iloc[0]
            memo_lines.append(f"   • Adjust schedules to add {int(peak['fallback_count']/20)+1} technicians during {peak['time_window']}")
            memo_lines.append(f"     Impact: Reduce travel by {(peak['avg_distance_km']-20)/peak['avg_distance_km']*100:.0f}%, improve {peak['fallback_count']} assignments")
        
        # Geographic relocation
        if cost_opportunities['high_distance_count'] > 50:
            memo_lines.append(f"   • Relocate or hire technicians in high-distance areas")
            memo_lines.append(f"     Impact: Save ~{potential_distance_savings/60:.0f} hours/week, ${potential_cost_savings:.0f}/week")
        
        memo_lines.append("\n3. STRATEGIC (Quarter 1-2):")
        memo_lines.append(f"   • Cross-train {len(under_utilized)} under-utilized technicians in shortage skills")
        memo_lines.append(f"   • Implement dynamic scheduling based on time-of-day demand patterns")
        memo_lines.append(f"   • Review and adjust capacity limits for consistently over-utilized technicians")
        
        # Footer
        memo_lines.append("\n" + "=" * 80)
        memo_lines.append("END OF STRATEGIC MEMO")
        memo_lines.append("=" * 80)
        
        memo = "\n".join(memo_lines)
        
        print("\n✅ Strategy memo generated successfully!")
        
        return memo
    
    # ============================================================
    # CONVENIENCE METHOD: RUN FULL ANALYSIS
    # ============================================================
    
    def analyze(self, save_to_file: bool = False, output_path: str = "/tmp/strategy_memo.txt") -> str:
        """
        Run complete strategic analysis and generate memo.
        
        Args:
            save_to_file: Whether to save memo to a file
            output_path: Path to save the memo
        
        Returns:
            str: Generated strategy memo
        """
        print("\n" + "=" * 80)
        print("STARTING STRATEGIC ANALYSIS")
        print("=" * 80)
        
        # Load data
        self.load_data()
        
        # Generate memo
        memo = self.generate_strategy_memo()
        
        # Print to console
        print("\n" + memo)
        
        # Save to file if requested
        if save_to_file:
            with open(output_path, 'w') as f:
                f.write(memo)
            print(f"\n✅ Memo saved to: {output_path}")
        
        return memo

# ============================================================
# USAGE EXAMPLE
# ============================================================

# Initialize the agent
strategy_agent = StrategyInsightsAgent(
    spark=spark,
    catalog="hackathon",
    schema="hackathon_fiber_vault"
)

# Run full analysis and generate memo
memo = strategy_agent.analyze(save_to_file=True, output_path="/tmp/dispatch_strategy_memo.txt")