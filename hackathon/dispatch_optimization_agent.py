# ============================================================
# DISPATCH OPTIMIZATION AGENT - ENHANCED VERSION
# With detailed constraint tracking and fallback assignments
# ============================================================

from pyspark.sql import DataFrame, functions as F, Window
from pyspark.sql.types import DoubleType, IntegerType, StringType
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION & CONSTANTS
# ============================================================

EARTH_RADIUS_KM = 6371.0

# Hard Constraints
HARD_MAX_DISTANCE_KM = 60.0

# Soft Constraint Weights
TRAVEL_BASE_COST = 0.5
TRAVEL_PENALTY_THRESHOLD_KM = 20.0
TRAVEL_PENALTY_MULTIPLIER = 2.0

UNDERUTIL_THRESHOLD = 0.4
OVERUTIL_THRESHOLD = 0.9
UNDERUTIL_PENALTY = 8.0
OVERUTIL_PENALTY = 15.0

SUCCESS_WEIGHT = 0.7
FTF_WEIGHT = 0.3
RISK_COST_MULTIPLIER = 10.0

DURATION_COST_FACTOR = 0.01

PRIORITY_WEIGHTS = {
    "Critical": 3.0,
    "High": 2.0,
    "Normal": 1.0,
    "Low": 0.5
}

# ============================================================
# DISPATCH OPTIMIZATION AGENT CLASS
# ============================================================

class DispatchOptimizationAgent:
    """
    Enhanced dispatch optimization agent with:
    - Detailed constraint tracking
    - Fallback assignments for failed hard constraints
    - Comprehensive optimization reasons
    """
    
    def __init__(
        self,
        spark,
        ml_agent,
        catalog: str = "hackathon",
        schema: str = "hackathon_fiber_vault"
    ):
        """Initialize the optimization agent."""
        self.spark = spark
        self.ml_agent = ml_agent
        self.catalog = catalog
        self.schema = schema
        self.table_prefix = f"{catalog}.{schema}"
        
        print("=" * 80)
        print("DISPATCH OPTIMIZATION AGENT - ENHANCED VERSION")
        print("=" * 80)
        print(f"Catalog: {catalog}")
        print(f"Schema: {schema}")
        print(f"Hard Max Distance: {HARD_MAX_DISTANCE_KM} km")
        print("NOTE: City/State constraint REMOVED")
        print("=" * 80)
    
    # ============================================================
    # DATA LOADING
    # ============================================================
    
    # ============================================================
    # UPDATED: Load data with date filtering
    # ============================================================

    def load_data(self) -> Tuple[DataFrame, DataFrame, DataFrame]:
        """Load all required tables and filter out dispatches before Nov 12."""
        print("\n[1/8] Loading data tables...")
        
        # Use self.dispatches_table if set, otherwise default
        dispatches_table = getattr(self, 'dispatches_table', 'current_dispatches_hackathon')
        
        self.dispatches_df = self.spark.table(f"{self.table_prefix}.{dispatches_table}")
        self.technicians_df = self.spark.table(f"{self.table_prefix}.technicians_hackathon")
        self.calendar_df = self.spark.table(f"{self.table_prefix}.technician_calendar_hackathon")
        
        # Filter out dispatches before November 12, 2025
        print(f"  Filtering dispatches to >= 2025-11-12...")
        original_count = self.dispatches_df.count()
        
        self.dispatches_df = self.dispatches_df.filter(
            F.to_date(F.col("Appointment_start_datetime")) >= F.lit("2025-11-12")
        )
        
        filtered_count = self.dispatches_df.count()
        removed_count = original_count - filtered_count
        
        print(f"  ✓ Dispatches: {filtered_count:,} (removed {removed_count:,} before Nov 12)")
        print(f"  ✓ Technicians: {self.technicians_df.count():,}")
        print(f"  ✓ Calendar: {self.calendar_df.count():,}")
        
        return self.dispatches_df, self.technicians_df, self.calendar_df
    
    # ============================================================
    # ENHANCED CANDIDATE GENERATION WITH CONSTRAINT TRACKING
    # ============================================================
    
    def generate_candidates_with_tracking(
        self,
        only_unoptimized: bool = True,
        only_pending: bool = False
    ) -> DataFrame:
        """
        Generate candidates with detailed constraint tracking.
        
        Constraint order:
        1. Date exists in calendar
        2. Skill match
        3. Capacity available
        4. Calendar availability
        5. Distance <= 60km
        
        Returns candidates with constraint_level:
        - 'ALL_CONSTRAINTS_MET': Passes all hard constraints
        - 'FAILED_DISTANCE': Failed only distance
        - 'FAILED_AVAILABILITY': Failed availability (passed skill, capacity)
        - 'FAILED_CAPACITY': Failed capacity (passed skill)
        - 'FAILED_SKILL': Failed skill match
        - 'NO_DATE_MATCH': No calendar date available
        """
        print("\n[2/8] Generating candidates with constraint tracking...")
        
        # Prepare dispatches
        dispatches = self.dispatches_df
        
        if only_unoptimized:
            dispatches = dispatches.filter(
                F.col("Optimized_technician_id").isNull() | 
                (F.col("Optimized_technician_id") == "")
            )
        
        if only_pending:
            dispatches = dispatches.filter(F.col("Status") == "Pending")
        
        initial_count = dispatches.count()
        print(f"  Starting dispatches: {initial_count:,}")
        
        dispatches = dispatches.withColumn(
            "Appointment_date",
            F.to_date(F.col("Appointment_start_datetime"))
        )
        
        # Generate ALL possible (dispatch, technician) pairs with constraint flags
        all_pairs = (
            dispatches.alias("d")
            .join(
                self.calendar_df.alias("cal"),
                (F.col("d.Appointment_date") == F.col("cal.Date")),
                "left"  # Left join to keep dispatches even if no date match
            )
            .join(
                self.technicians_df.alias("t"),
                (F.col("cal.Technician_id") == F.col("t.Technician_id")),
                "left"  # Left join to keep all combinations
            )
        )
        
        # Add constraint check flags
        all_pairs = all_pairs.withColumn(
            "has_date_match",
            F.when(F.col("cal.Date").isNotNull(), 1).otherwise(0)
        ).withColumn(
            "skill_match",
            F.when(F.col("d.Required_skill") == F.col("t.Primary_skill"), 1).otherwise(0)
        ).withColumn(
            "capacity_available",
            F.when(F.col("t.Current_assignments") < F.col("t.Workload_capacity"), 1).otherwise(0)
        ).withColumn(
            "calendar_available",
            F.when(F.col("cal.Available") == 1, 1).otherwise(0)
        )
        
        # Calculate distance
        all_pairs = all_pairs.withColumn(
            "distance_km",
            F.when(
                F.col("t.Latitude").isNotNull() & F.col("d.Customer_latitude").isNotNull(),
                F.lit(2) * F.lit(EARTH_RADIUS_KM) * F.asin(
                    F.sqrt(
                        F.pow(F.sin(F.radians(F.col("d.Customer_latitude") - F.col("t.Latitude")) / 2), 2) +
                        F.cos(F.radians(F.col("t.Latitude"))) * 
                        F.cos(F.radians(F.col("d.Customer_latitude"))) *
                        F.pow(F.sin(F.radians(F.col("d.Customer_longitude") - F.col("t.Longitude")) / 2), 2)
                    )
                )
            ).otherwise(9999.0)  # Large value for missing coords
        ).withColumn(
            "distance_ok",
            F.when(F.col("distance_km") <= HARD_MAX_DISTANCE_KM, 1).otherwise(0)
        )
        
        # Determine constraint level (ordered by priority)
        all_pairs = all_pairs.withColumn(
            "constraint_level",
            F.when(F.col("has_date_match") == 0, "NO_DATE_MATCH")
            .when(F.col("skill_match") == 0, "FAILED_SKILL")
            .when(F.col("capacity_available") == 0, "FAILED_CAPACITY")
            .when(F.col("calendar_available") == 0, "FAILED_AVAILABILITY")
            .when(F.col("distance_ok") == 0, "FAILED_DISTANCE")
            .otherwise("ALL_CONSTRAINTS_MET")
        ).withColumn(
            "constraints_passed",
            F.col("has_date_match") + F.col("skill_match") + F.col("capacity_available") + 
            F.col("calendar_available") + F.col("distance_ok")
        )
        
        # Select relevant columns
        candidates = all_pairs.select(
            # Dispatch info
            F.col("d.Dispatch_id").alias("Dispatch_id"),
            F.col("d.Ticket_type").alias("Ticket_type"),
            F.col("d.Order_type").alias("Order_type"),
            F.col("d.Required_skill").alias("Required_skill"),
            F.col("d.Priority").alias("Priority"),
            F.col("d.Duration_min").alias("Duration_min"),
            F.col("d.Customer_latitude").alias("Customer_latitude"),
            F.col("d.Customer_longitude").alias("Customer_longitude"),
            F.col("d.City").alias("Dispatch_city"),
            F.col("d.State").alias("Dispatch_state"),
            F.col("d.Appointment_date").alias("Appointment_date"),
            F.col("d.Appointment_start_datetime").alias("Appointment_start_datetime"),
            F.col("d.Appointment_end_datetime").alias("Appointment_end_datetime"),
            F.col("d.Assigned_technician_id").alias("Baseline_technician_id"),
            
            # Technician info
            F.col("t.Technician_id").alias("Technician_id"),
            F.col("t.Name").alias("Technician_name"),
            F.col("t.City").alias("Tech_city"),
            F.col("t.State").alias("Tech_state"),
            F.col("t.Latitude").alias("Tech_latitude"),
            F.col("t.Longitude").alias("Tech_longitude"),
            F.col("t.Primary_skill").alias("Tech_skill"),
            F.col("t.Current_assignments").alias("Tech_current_assignments"),
            F.col("t.Workload_capacity").alias("Tech_workload_capacity"),
            
            # Calendar info
            F.col("cal.Max_assignments").alias("Tech_max_assignments"),
            F.col("cal.Start_time").alias("Tech_start_time"),
            F.col("cal.End_time").alias("Tech_end_time"),
            
            # Constraint tracking
            "distance_km",
            "constraint_level",
            "constraints_passed",
            "skill_match",
            "capacity_available",
            "calendar_available",
            "distance_ok"
        ).filter(
            F.col("Technician_id").isNotNull()  # Must have at least some technician match
        )
        
        # Add utilization
        candidates = candidates.withColumn(
            "tech_current_utilization",
            F.when(
                F.col("Tech_workload_capacity") > 0,
                F.col("Tech_current_assignments") / F.col("Tech_workload_capacity")
            ).otherwise(0.0)
        )
        
        # Report statistics
        total_pairs = candidates.count()
        all_met = candidates.filter(F.col("constraint_level") == "ALL_CONSTRAINTS_MET").count()
        unique_dispatches_all_met = candidates.filter(
            F.col("constraint_level") == "ALL_CONSTRAINTS_MET"
        ).select("Dispatch_id").distinct().count()
        
        print(f"  Total candidate pairs: {total_pairs:,}")
        print(f"  Pairs meeting ALL constraints: {all_met:,}")
        print(f"  Unique dispatches with ≥1 feasible tech: {unique_dispatches_all_met:,}")
        
        # Constraint failure breakdown
        print("\n  Constraint failure breakdown (unique dispatches):")
        constraint_summary = candidates.groupBy("constraint_level").agg(
            F.countDistinct("Dispatch_id").alias("unique_dispatches")
        ).orderBy(F.desc("unique_dispatches"))
        constraint_summary.show(truncate=False)
        
        return candidates
    
    # ============================================================
    # ML SCORING
    # ============================================================
    
    def score_candidates_with_ml(self, candidates_df: DataFrame) -> pd.DataFrame:
        """Apply ML model to score candidates."""
        print("\n[3/8] Scoring candidates with ML model...")
        
        candidates_pd = candidates_df.toPandas()
        print(f"  Converting {len(candidates_pd):,} pairs to Pandas...")
        
        # Column mapping
        column_mapping = {
            'Dispatch_id': 'dispatch_id',
            'Ticket_type': 'ticket_type',
            'Order_type': 'order_type',
            'Technician_id': 'technician_id',
            'Tech_latitude': 'technician_latitude',
            'Tech_longitude': 'technician_longitude',
            'Customer_latitude': 'customer_latitude',
            'Customer_longitude': 'customer_longitude',
            'Tech_city': 'technician_city',
            'Dispatch_city': 'dispatch_city',
            'Tech_state': 'technician_state',
            'Dispatch_state': 'dispatch_state',
            'Tech_skill': 'technician_skill',
            'Required_skill': 'required_skill',
            'Priority': 'priority',
            'Duration_min': 'duration_min',
            'Tech_current_assignments': 'technician_current_assignments',
            'Tech_workload_capacity': 'technician_workload_capacity',
            'Appointment_start_datetime': 'appointment_start_time',
            'Appointment_date': 'appointment_date'
        }
        
        candidates_pd_renamed = candidates_pd.rename(columns={
            k: v for k, v in column_mapping.items() if k in candidates_pd.columns
        })
        
        # Score using ML agent
        scored_pd = self.ml_agent.score_candidates(candidates_pd_renamed)
        
        # Merge back
        result_pd = candidates_pd.merge(
            scored_pd[['dispatch_id', 'technician_id', 'p_productive', 'p_ftf']],
            left_on=['Dispatch_id', 'Technician_id'],
            right_on=['dispatch_id', 'technician_id'],
            how='left'
        ).drop(columns=['dispatch_id', 'technician_id'], errors='ignore')
        
        result_pd['p_productive'] = result_pd['p_productive'].fillna(0.5)
        result_pd['p_ftf'] = result_pd['p_ftf'].fillna(0.5)
        
        print(f"  ✓ ML scoring complete")
        
        return result_pd
    
    # ============================================================
    # SOFT CONSTRAINT SCORING
    # ============================================================
    
    def calculate_soft_costs(self, candidates_pd: pd.DataFrame) -> pd.DataFrame:
        """Calculate soft constraint costs."""
        print("\n[4/8] Calculating soft constraint costs...")
        
        # Travel cost
        candidates_pd['travel_cost'] = (
            TRAVEL_BASE_COST * candidates_pd['distance_km'] +
            TRAVEL_PENALTY_MULTIPLIER * np.maximum(0, candidates_pd['distance_km'] - TRAVEL_PENALTY_THRESHOLD_KM)
        )
        
        # Workload balance
        util = candidates_pd['tech_current_utilization']
        candidates_pd['workload_cost'] = np.where(
            util < UNDERUTIL_THRESHOLD,
            UNDERUTIL_PENALTY * (UNDERUTIL_THRESHOLD - util),
            np.where(
                util > OVERUTIL_THRESHOLD,
                OVERUTIL_PENALTY * (util - OVERUTIL_THRESHOLD),
                0.0
            )
        )
        
        # Quality score
        candidates_pd['quality_score'] = (
            SUCCESS_WEIGHT * candidates_pd['p_productive'] +
            FTF_WEIGHT * candidates_pd['p_ftf']
        )
        candidates_pd['success_risk_cost'] = (
            RISK_COST_MULTIPLIER * (1.0 - candidates_pd['quality_score'])
        )
        
        # Duration cost
        candidates_pd['duration_cost'] = DURATION_COST_FACTOR * candidates_pd['Duration_min']
        
        # Base job cost
        candidates_pd['base_job_cost'] = (
            candidates_pd['travel_cost'] +
            candidates_pd['workload_cost'] +
            candidates_pd['success_risk_cost'] +
            candidates_pd['duration_cost']
        )
        
        # Priority weighting
        candidates_pd['priority_weight'] = candidates_pd['Priority'].map(PRIORITY_WEIGHTS).fillna(1.0)
        candidates_pd['final_cost'] = candidates_pd['base_job_cost'] * candidates_pd['priority_weight']
        
        # Add penalty for constraint violations (for fallback assignments)
        candidates_pd['constraint_penalty'] = (5 - candidates_pd['constraints_passed']) * 100.0
        candidates_pd['final_cost_with_penalty'] = candidates_pd['final_cost'] + candidates_pd['constraint_penalty']
        
        # Score (negative cost)
        candidates_pd['final_score'] = -candidates_pd['final_cost']
        candidates_pd['final_score_with_penalty'] = -candidates_pd['final_cost_with_penalty']
        
        print(f"  ✓ Cost calculation complete")
        
        return candidates_pd
    
    # ============================================================
    # OPTIMAL ASSIGNMENT SELECTION
    # ============================================================
    
    def select_optimal_assignments(self, scored_candidates_pd: pd.DataFrame) -> pd.DataFrame:
        """
        Select best technician per dispatch.
        Priority: All constraints met > Most constraints met > Best score
        """
        print("\n[5/8] Selecting optimal assignments...")
        
        # Sort by: constraint_level priority, then score
        constraint_priority = {
            'ALL_CONSTRAINTS_MET': 1,
            'FAILED_DISTANCE': 2,
            'FAILED_AVAILABILITY': 3,
            'FAILED_CAPACITY': 4,
            'FAILED_SKILL': 5,
            'NO_DATE_MATCH': 6
        }
        
        scored_candidates_pd['constraint_priority'] = scored_candidates_pd['constraint_level'].map(constraint_priority)
        
        # Sort: Best constraint level first, then best score
        sorted_candidates = scored_candidates_pd.sort_values(
            ['Dispatch_id', 'constraint_priority', 'final_score'],
            ascending=[True, True, False]
        )
        
        # Get best per dispatch
        best_assignments = sorted_candidates.groupby('Dispatch_id').first().reset_index()
        
        all_constraints = (best_assignments['constraint_level'] == 'ALL_CONSTRAINTS_MET').sum()
        fallback = len(best_assignments) - all_constraints
        
        print(f"  ✓ Optimal assignments selected: {len(best_assignments):,}")
        print(f"    - Meeting ALL constraints: {all_constraints}")
        print(f"    - Fallback assignments: {fallback}")
        
        return best_assignments
    
    # ============================================================
    # RESULT TABLE GENERATION WITH DETAILED REASONS
    # ============================================================
    
    # ============================================================
    # UPDATED: Create optimized table with Optimization_status
    # ============================================================

    def create_optimized_table(self, optimal_assignments_pd: pd.DataFrame) -> DataFrame:
        """Create final table with Optimization_status updated."""
        print("\n[6/8] Creating optimized dispatches table...")
        
        optimal_spark = self.spark.createDataFrame(optimal_assignments_pd)
        
        # Join with original
        optimized_full = (
            self.dispatches_df.alias("orig")
            .join(
                optimal_spark.alias("opt"),
                F.col("orig.Dispatch_id") == F.col("opt.Dispatch_id"),
                "left"
            )
        )
        
        # Create result columns
        optimized_full = optimized_full.select(
            # Original columns
            F.col("orig.Dispatch_id"),
            F.col("orig.Ticket_type"),
            F.col("orig.Order_type"),
            F.col("orig.Priority"),
            F.col("orig.Required_skill"),
            F.col("orig.Status"),
            F.col("orig.Street"),
            F.col("orig.City"),
            F.col("orig.County"),
            F.col("orig.State"),
            F.col("orig.Postal_code"),
            F.col("orig.Customer_latitude"),
            F.col("orig.Customer_longitude"),
            F.col("orig.Appointment_start_datetime"),
            F.col("orig.Appointment_end_datetime"),
            F.col("orig.Duration_min"),
            F.col("orig.Assigned_technician_id"),  # Baseline unchanged
            F.col("orig.Resolution_type"),
            
            # Optimized columns
            F.coalesce(F.col("opt.Technician_id"), F.col("orig.Optimized_technician_id")).alias("Optimized_technician_id"),
            F.col("opt.Technician_name").alias("Optimized_technician_name"),
            F.col("opt.constraint_level").alias("Optimization_constraint_status"),
            F.col("opt.constraints_passed").alias("Optimization_constraints_passed"),
            F.col("opt.final_score").alias("Optimized_score"),
            F.col("opt.distance_km").alias("Optimized_distance_km"),
            F.col("opt.tech_current_utilization").alias("Optimized_utilization"),
            F.col("opt.p_productive").alias("Optimized_p_productive"),
            F.col("opt.p_ftf").alias("Optimized_p_ftf"),
            F.col("opt.quality_score").alias("Optimized_quality_score"),
            F.col("opt.travel_cost").alias("Optimized_travel_cost"),
            F.col("opt.workload_cost").alias("Optimized_workload_cost"),
            F.col("opt.success_risk_cost").alias("Optimized_success_risk_cost"),
            F.col("opt.duration_cost").alias("Optimized_duration_cost"),
            F.col("opt.priority_weight").alias("Optimized_priority_weight"),
            F.col("opt.skill_match").alias("Optimization_skill_match"),
            F.col("opt.capacity_available").alias("Optimization_capacity_available"),
            F.col("opt.calendar_available").alias("Optimization_calendar_available"),
            F.col("opt.distance_ok").alias("Optimization_distance_ok"),
            
            # UPDATE Optimization_status: "Complete" if assigned, otherwise keep original or "Pending"
            F.when(
                F.col("opt.Technician_id").isNotNull(),
                F.lit("Complete")
            ).otherwise(
                F.coalesce(F.col("orig.Optimization_status"), F.lit("Pending"))
            ).alias("Optimization_status"),
            
            # Timestamp
            F.when(
                F.col("opt.Technician_id").isNotNull(),
                F.current_timestamp()
            ).otherwise(F.col("orig.Optimization_timestamp")).alias("Optimization_timestamp"),
            
            F.col("orig.Optimization_confidence")
        )
        
        # Generate detailed optimization reason
        optimized_full = optimized_full.withColumn(
            "Optimization_reason",
            F.when(
                F.col("Optimization_constraint_status") == "ALL_CONSTRAINTS_MET",
                F.concat(
                    F.lit("✓ ALL CONSTRAINTS MET | Assigned: "),
                    F.col("Optimized_technician_name"),
                    F.lit(" ("),
                    F.col("Optimized_technician_id"),
                    F.lit(") | Distance: "),
                    F.round(F.col("Optimized_distance_km"), 1),
                    F.lit(" km | Utilization: "),
                    F.round(F.col("Optimized_utilization") * 100, 0),
                    F.lit("% | ML Success: "),
                    F.round(F.col("Optimized_p_productive") * 100, 0),
                    F.lit("% | ML FTF: "),
                    F.round(F.col("Optimized_p_ftf") * 100, 0),
                    F.lit("% | Score: "),
                    F.round(F.col("Optimized_score"), 1)
                )
            ).when(
                F.col("Optimization_constraint_status") == "FAILED_DISTANCE",
                F.concat(
                    F.lit("⚠ FALLBACK (Distance > 60km) | Best available: "),
                    F.col("Optimized_technician_name"),
                    F.lit(" | Distance: "),
                    F.round(F.col("Optimized_distance_km"), 1),
                    F.lit(" km | Passed: Skill, Capacity, Availability")
                )
            ).when(
                F.col("Optimization_constraint_status") == "FAILED_AVAILABILITY",
                F.concat(
                    F.lit("⚠ FALLBACK (Technician unavailable on date) | Best available: "),
                    F.col("Optimized_technician_name"),
                    F.lit(" | Passed: Skill, Capacity | Failed: Availability")
                )
            ).when(
                F.col("Optimization_constraint_status") == "FAILED_CAPACITY",
                F.concat(
                    F.lit("⚠ FALLBACK (Technician over capacity) | Best available: "),
                    F.col("Optimized_technician_name"),
                    F.lit(" | Passed: Skill | Failed: Capacity")
                )
            ).when(
                F.col("Optimization_constraint_status") == "FAILED_SKILL",
                F.concat(
                    F.lit("⚠ FALLBACK (No skill match) | Best available: "),
                    F.col("Optimized_technician_name"),
                    F.lit(" | Required: "),
                    F.col("Required_skill"),
                    F.lit(" | Tech has: Different skill")
                )
            ).when(
                F.col("Optimization_constraint_status") == "NO_DATE_MATCH",
                F.lit("✗ NO ASSIGNMENT | Calendar date unavailable for this appointment date")
            ).otherwise(
                F.lit("✗ NO ASSIGNMENT | No technicians available")
            )
        )
        
        # Add helper flag for all constraints met (keep this, it's useful separate from status)
        optimized_full = optimized_full.withColumn(
            "All_constraints_met",
            F.when(F.col("Optimization_constraint_status") == "ALL_CONSTRAINTS_MET", F.lit(True)).otherwise(F.lit(False))
        )
        
        print(f"  ✓ Optimized table created")
        
        return optimized_full
    
    # ============================================================
    # WRITE TO TABLE
    # ============================================================
    
    def write_optimized_table(
        self,
        optimized_df: DataFrame,
        table_name: str = "current_dispatches_hackathon_opt",
        mode: str = "overwrite"
    ):
        """Write optimized dispatches to Delta table."""
        full_table_name = f"{self.table_prefix}.{table_name}"
        print(f"\n[7/8] Writing to table: {full_table_name}")
        
        optimized_df.write.format("delta").mode(mode).saveAsTable(full_table_name)
        
        print(f"  ✓ Successfully wrote to {full_table_name}")
    
    # ============================================================
    # MAIN OPTIMIZATION WORKFLOW
    # ============================================================
    
    def optimize(
        self,
        only_unoptimized: bool = True,
        only_pending: bool = False,
        write_to_table: bool = True,
        table_name: str = "current_dispatches_hackathon_opt"
    ) -> DataFrame:
        """Run complete optimization workflow."""
        print("\n" + "=" * 80)
        print("STARTING ENHANCED DISPATCH OPTIMIZATION")
        print("=" * 80)
        
        # Load data (now with date filtering)
        self.load_data()
        
        # Generate candidates with tracking
        candidates_df = self.generate_candidates_with_tracking(only_unoptimized, only_pending)
        
        if candidates_df.count() == 0:
            print("\n❌ No candidates found")
            return None
        
        # Score with ML
        scored_candidates_pd = self.score_candidates_with_ml(candidates_df)
        
        # Calculate costs
        scored_candidates_pd = self.calculate_soft_costs(scored_candidates_pd)
        
        # Select optimal
        optimal_assignments_pd = self.select_optimal_assignments(scored_candidates_pd)
        
        # Create final table
        optimized_df = self.create_optimized_table(optimal_assignments_pd)
        
        # Write
        if write_to_table:
            self.write_optimized_table(optimized_df, table_name)
        
        # Summary
        print("\n[8/8] Optimization Summary")
        print("=" * 80)
        
        total = optimized_df.count()
        complete = optimized_df.filter(F.col("Optimization_status") == "Complete").count()
        all_met = optimized_df.filter(F.col("All_constraints_met") == True).count()
        fallback = complete - all_met
        pending = total - complete
        
        print(f"Total dispatches (>= Nov 12):  {total:>6}")
        print(f"Optimization_status = Complete: {complete:>6} ({complete/total*100:.1f}%)")
        print(f"  - All constraints met:        {all_met:>6} ({all_met/total*100:.1f}%)")
        print(f"  - Fallback assignments:       {fallback:>6} ({fallback/total*100:.1f}%)")
        print(f"Optimization_status = Pending:  {pending:>6} ({pending/total*100:.1f}%)")
        print("=" * 80)
        
        return optimized_df