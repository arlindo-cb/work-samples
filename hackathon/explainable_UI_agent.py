# ============================================================
# COMPLETE DISPATCH TICKETING UI WITH EXPLAINABLE AI & INSIGHTS
# ============================================================

from pyspark.sql.types import *
import pyspark.sql.functions as F
from datetime import datetime
import ipywidgets as widgets
from IPython.display import display, HTML
import pandas as pd
import sys

sys.path.append('/Workspace/Users/Hackathon_Fiber_Vault')
from success_model_agent import SuccessModelAgent
from dispatch_optimization_agent import DispatchOptimizationAgent
from strategy_insights_agent import StrategyInsightsAgent  

# ============================================================
# EXPLAINABLE AI: REASONING GENERATOR
# ============================================================

def generate_assignment_explanation(result_row, all_candidates_df=None):
    """
    Generate human-readable explanation for why a technician was assigned.
    
    Args:
        result_row: Pandas Series with the selected assignment details
        all_candidates_df: Optional DataFrame with all candidates for comparison
    
    Returns:
        str: Human-readable explanation
    """
    tech_id = result_row.get('Technician_id')
    tech_name = result_row.get('Technician_name', 'Unknown')
    distance = result_row.get('distance_km', 0)
    utilization = result_row.get('tech_current_utilization', 0)
    skill_match = result_row.get('skill_match', 0)
    p_productive = result_row.get('p_productive', 0)
    p_ftf = result_row.get('p_ftf', 0)
    quality_score = result_row.get('quality_score', 0)
    constraint_level = result_row.get('constraint_level', '')
    
    # Build explanation parts
    parts = []
    
    # 1. Primary decision factors
    parts.append(f"🎯 **Primary Selection Factors:**")
    
    if skill_match == 1:
        parts.append(f"   ✓ **Skill Match**: Technician has the required skill")
    else:
        parts.append(f"   ⚠ **Skill Mismatch**: Technician assigned despite different skill (fallback)")
    
    parts.append(f"   ✓ **Proximity**: {distance:.1f} km from customer location")
    
    # Distance assessment
    if distance <= 10:
        parts.append(f"      → Excellent proximity (within 10 km)")
    elif distance <= 20:
        parts.append(f"      → Good proximity (10-20 km)")
    elif distance <= 40:
        parts.append(f"      → Moderate distance (20-40 km)")
    else:
        parts.append(f"      → Far distance (>40 km, may incur higher travel cost)")
    
    # 2. Workload & Capacity
    parts.append(f"\n📊 **Workload & Availability:**")
    parts.append(f"   ✓ **Current Utilization**: {utilization*100:.0f}% of daily capacity")
    
    if utilization < 0.4:
        parts.append(f"      → Low utilization - technician has significant availability")
    elif utilization < 0.7:
        parts.append(f"      → Balanced workload - optimal assignment")
    elif utilization < 0.9:
        parts.append(f"      → High utilization but still available")
    else:
        parts.append(f"      → Near/at capacity - may be stretched thin")
    
    # 3. AI-Predicted Success Metrics
    parts.append(f"\n🤖 **AI-Predicted Success Metrics:**")
    parts.append(f"   ✓ **Productive Dispatch Probability**: {p_productive*100:.1f}%")
    parts.append(f"   ✓ **First-Time-Fix Probability**: {p_ftf*100:.1f}%")
    parts.append(f"   ✓ **Overall Quality Score**: {quality_score:.3f}")
    
    if quality_score >= 0.8:
        parts.append(f"      → Excellent predicted outcome")
    elif quality_score >= 0.6:
        parts.append(f"      → Good predicted outcome")
    else:
        parts.append(f"      → Fair predicted outcome")
    
    # 4. Cost Components (if available)
    travel_cost = result_row.get('travel_cost', 0)
    workload_cost = result_row.get('workload_cost', 0)
    success_risk_cost = result_row.get('success_risk_cost', 0)
    
    if travel_cost or workload_cost or success_risk_cost:
        parts.append(f"\n💰 **Optimization Cost Analysis:**")
        parts.append(f"   • Travel Cost: {travel_cost:.2f}")
        parts.append(f"   • Workload Balance Cost: {workload_cost:.2f}")
        parts.append(f"   • Success Risk Cost: {success_risk_cost:.2f}")
    
    # 5. Constraint Status
    parts.append(f"\n✅ **Constraint Validation:**")
    if constraint_level == 'ALL_CONSTRAINTS_MET':
        parts.append(f"   All hard constraints satisfied:")
        parts.append(f"   • Skill requirement: ✓")
        parts.append(f"   • Workload capacity: ✓")
        parts.append(f"   • Calendar availability: ✓")
        parts.append(f"   • Distance limit (≤60 km): ✓")
    else:
        constraints_passed = result_row.get('constraints_passed', 0)
        parts.append(f"   ⚠ Fallback assignment ({constraints_passed}/5 constraints met)")
        parts.append(f"   Status: {constraint_level}")
    
    # 6. Comparison to alternatives (if available)
    if all_candidates_df is not None and len(all_candidates_df) > 1:
        # Get top 3 candidates
        top_candidates = all_candidates_df.nlargest(3, 'final_score')
        
        if len(top_candidates) > 1:
            parts.append(f"\n🔍 **Comparison to Alternatives:**")
            parts.append(f"   This technician ranked #1 among {len(all_candidates_df)} candidates")
            
            # Show why this was better than #2
            if len(top_candidates) >= 2:
                alt = top_candidates.iloc[1]
                alt_name = alt.get('Technician_name', 'Unknown')
                alt_dist = alt.get('distance_km', 999)
                alt_util = alt.get('tech_current_utilization', 0)
                alt_score = alt.get('final_score', 0)
                
                parts.append(f"   vs. #{2}: {alt_name}")
                
                if distance < alt_dist:
                    parts.append(f"      → {distance:.1f} km vs {alt_dist:.1f} km (closer)")
                if utilization < alt_util and utilization < 0.9:
                    parts.append(f"      → {utilization*100:.0f}% vs {alt_util*100:.0f}% utilization (better availability)")
                if result_row.get('final_score', 0) > alt_score:
                    parts.append(f"      → Higher optimization score ({result_row.get('final_score', 0):.1f} vs {alt_score:.1f})")
    
    # 7. Final recommendation
    parts.append(f"\n📝 **Final Recommendation:**")
    if constraint_level == 'ALL_CONSTRAINTS_MET':
        if quality_score >= 0.7 and distance <= 20 and utilization < 0.8:
            parts.append(f"   ✓ **OPTIMAL**: All criteria met with excellent metrics")
        elif quality_score >= 0.6:
            parts.append(f"   ✓ **GOOD**: All constraints met, solid performance expected")
        else:
            parts.append(f"   ✓ **ACCEPTABLE**: All constraints met, moderate performance expected")
    else:
        parts.append(f"   ⚠ **FALLBACK**: Best available option given constraint violations")
    
    return "\n".join(parts)

# ============================================================
# BACKEND FUNCTION WITH EXPLAINABLE AI
# ============================================================

def submit_dispatch_ticket(spark, ticket_data):
    """Backend function with automatic optimization and explainable AI."""
    try:
        # STEP 1: Create dispatch
        max_id_result = spark.sql("""
            SELECT COALESCE(MAX(Dispatch_id), 200000000) as max_id 
            FROM hackathon.hackathon_fiber_vault.current_dispatches_hackathon_ui
        """).collect()
        
        new_dispatch_id = max_id_result[0]['max_id'] + 1
        appt_start = datetime.strptime(ticket_data['appointment_start'], '%Y-%m-%dT%H:%M')
        appt_end = datetime.strptime(ticket_data['appointment_end'], '%Y-%m-%dT%H:%M')
        
        # Explicit schema
        schema = StructType([
            StructField("Dispatch_id", LongType(), False),
            StructField("Ticket_type", StringType(), True),
            StructField("Order_type", StringType(), True),
            StructField("Priority", StringType(), True),
            StructField("Required_skill", StringType(), True),
            StructField("Status", StringType(), True),
            StructField("Street", StringType(), True),
            StructField("City", StringType(), True),
            StructField("County", StringType(), True),
            StructField("State", StringType(), True),
            StructField("Postal_code", LongType(), True),
            StructField("Customer_latitude", DoubleType(), True),
            StructField("Customer_longitude", DoubleType(), True),
            StructField("Appointment_start_datetime", TimestampType(), True),
            StructField("Appointment_end_datetime", TimestampType(), True),
            StructField("Duration_min", LongType(), True),
            StructField("Assigned_technician_id", StringType(), True),
            StructField("Resolution_type", StringType(), True),
            StructField("Optimized_technician_id", StringType(), True),
            StructField("Optimized_technician_name", StringType(), True),
            StructField("Optimization_constraint_status", StringType(), True),
            StructField("Optimization_constraints_passed", IntegerType(), True),
            StructField("Optimized_score", DoubleType(), True),
            StructField("Optimized_distance_km", DoubleType(), True),
            StructField("Optimized_utilization", DoubleType(), True),
            StructField("Optimized_p_productive", DoubleType(), True),
            StructField("Optimized_p_ftf", DoubleType(), True),
            StructField("Optimized_quality_score", DoubleType(), True),
            StructField("Optimized_travel_cost", DoubleType(), True),
            StructField("Optimized_workload_cost", DoubleType(), True),
            StructField("Optimized_success_risk_cost", DoubleType(), True),
            StructField("Optimized_duration_cost", DoubleType(), True),
            StructField("Optimized_priority_weight", DoubleType(), True),
            StructField("Optimization_skill_match", IntegerType(), True),
            StructField("Optimization_capacity_available", IntegerType(), True),
            StructField("Optimization_calendar_available", IntegerType(), True),
            StructField("Optimization_distance_ok", IntegerType(), True),
            StructField("Optimization_status", StringType(), True),
            StructField("Optimization_timestamp", TimestampType(), True),
            StructField("Optimization_confidence", StringType(), True),
            StructField("Optimization_reason", StringType(), True),
            StructField("All_constraints_met", BooleanType(), True)
        ])
        
        new_row = (
            int(new_dispatch_id), str(ticket_data['ticket_type']), str(ticket_data['order_type']),
            str(ticket_data['priority']), str(ticket_data['required_skill']), 'pending',
            str(ticket_data['street']), str(ticket_data['city']), '', str(ticket_data['state']),
            int(ticket_data['postal_code']), float(ticket_data['customer_latitude']),
            float(ticket_data['customer_longitude']), appt_start, appt_end,
            int(ticket_data['duration_min']), None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None, None, None,
            'Pending', None, None, 'Pending', False
        )
        
        new_dispatch_df = spark.createDataFrame([new_row], schema)
        new_dispatch_df.write.mode("append").saveAsTable("hackathon.hackathon_fiber_vault.current_dispatches_hackathon_ui")
        
        print(f"✅ Dispatch {new_dispatch_id} created\n")
        
        # STEP 2-5: Run optimization
        print("🔄 Loading ML agent...")
        ml_agent = SuccessModelAgent(spark, config={'catalog': 'hackathon', 'schema': 'hackathon_fiber_vault'})
        ml_agent.load_models_from_table()
        
        print("🔄 Initializing optimization agent...")
        opt_agent = DispatchOptimizationAgent(
            spark=spark, ml_agent=ml_agent, catalog="hackathon", schema="hackathon_fiber_vault"
        )
        
        opt_agent.dispatches_table = "current_dispatches_hackathon_ui"
        opt_agent.load_data()
        opt_agent.dispatches_df = opt_agent.dispatches_df.filter(F.col("Dispatch_id") == new_dispatch_id)
        
        print("🔄 Running optimization pipeline...")
        candidates_df = opt_agent.generate_candidates_with_tracking()
        scored_df = opt_agent.score_candidates_with_ml(candidates_df)
        costed_df = opt_agent.calculate_soft_costs(scored_df)
        optimal_df = opt_agent.select_optimal_assignments(costed_df)
        
        # STEP 6: Update the record and generate explanation
        print("🔄 Updating dispatch record...")
        
        if len(optimal_df) > 0:
            result_row = optimal_df[optimal_df['Dispatch_id'] == new_dispatch_id].iloc[0]
            
            # Get all candidates for comparison in explanation
            all_candidates_for_dispatch = costed_df[costed_df['Dispatch_id'] == new_dispatch_id]
            
            # Generate explainable AI reasoning
            print("🧠 Generating assignment explanation...")
            ai_explanation = generate_assignment_explanation(result_row, all_candidates_for_dispatch)
            
            tech_id = result_row.get('Technician_id')
            tech_name = str(result_row.get('Technician_name', 'Unknown')).replace("'", "''")
            constraint_status = str(result_row.get('constraint_level', '')).replace("'", "''")
            constraints_passed = int(result_row.get('constraints_passed', 0))
            all_constraints_met = bool(result_row.get('constraint_level') == 'ALL_CONSTRAINTS_MET')
            
            if all_constraints_met:
                opt_reason = f"Optimal: All constraints met, score={result_row.get('final_score', 0):.2f}"
            else:
                opt_reason = f"Partial: {constraint_status}, {constraints_passed}/4 constraints"
            opt_reason = opt_reason.replace("'", "''")
            
            update_sql = f"""
            UPDATE hackathon.hackathon_fiber_vault.current_dispatches_hackathon_ui
            SET 
                Optimized_technician_id = '{tech_id}',
                Optimized_technician_name = '{tech_name}',
                Optimization_constraint_status = '{constraint_status}',
                Optimization_constraints_passed = {constraints_passed},
                Optimized_score = {float(result_row.get('final_score', 0))},
                Optimized_distance_km = {float(result_row.get('distance_km', 0))},
                Optimized_utilization = {float(result_row.get('tech_current_utilization', 0))},
                Optimized_p_productive = {float(result_row.get('p_productive', 0))},
                Optimized_p_ftf = {float(result_row.get('p_ftf', 0))},
                Optimized_quality_score = {float(result_row.get('quality_score', 0))},
                Optimized_travel_cost = {float(result_row.get('travel_cost', 0))},
                Optimized_workload_cost = {float(result_row.get('workload_cost', 0))},
                Optimized_success_risk_cost = {float(result_row.get('success_risk_cost', 0))},
                Optimized_duration_cost = {float(result_row.get('duration_cost', 0))},
                Optimized_priority_weight = {float(result_row.get('priority_weight', 1.0))},
                Optimization_skill_match = {int(result_row.get('skill_match', 0))},
                Optimization_capacity_available = {int(result_row.get('capacity_available', 0))},
                Optimization_calendar_available = {int(result_row.get('calendar_available', 0))},
                Optimization_distance_ok = {int(result_row.get('distance_ok', 0))},
                Optimization_status = 'Complete',
                Optimization_reason = '{opt_reason}',
                All_constraints_met = {all_constraints_met}
            WHERE Dispatch_id = {new_dispatch_id}
            """
            
            spark.sql(update_sql)
            
            return {
                'success': True,
                'dispatch_id': new_dispatch_id,
                'technician_id': tech_id,
                'technician_name': tech_name,
                'distance_km': float(result_row.get('distance_km', 0)),
                'p_productive': float(result_row.get('p_productive', 0)),
                'p_ftf': float(result_row.get('p_ftf', 0)),
                'all_constraints_met': all_constraints_met,
                'reason': opt_reason,
                'ai_explanation': ai_explanation
            }
        else:
            update_sql = f"""
            UPDATE hackathon.hackathon_fiber_vault.current_dispatches_hackathon_ui
            SET 
                Optimization_status = 'Complete',
                Optimization_reason = 'No feasible technicians found',
                All_constraints_met = false
            WHERE Dispatch_id = {new_dispatch_id}
            """
            spark.sql(update_sql)
            
            return {
                'success': True,
                'dispatch_id': new_dispatch_id,
                'technician_id': None,
                'technician_name': 'None',
                'distance_km': 0,
                'p_productive': 0,
                'p_ftf': 0,
                'all_constraints_met': False,
                'reason': 'No feasible technicians found',
                'ai_explanation': '✗ No technicians available meeting minimum criteria for this dispatch.'
            }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'dispatch_id': None,
            'message': f"Error: {str(e)}"
        }

# ============================================================
# UI CLASS (with explainable AI display & insights button)
# ============================================================

class DispatchTicketingUI:
    """Interactive UI for dispatch ticket submission."""
    
    def __init__(self, spark):
        self.spark = spark
        self._create_widgets()
        
    def _create_widgets(self):
        """Create all UI widgets with Frontier styling and prefilled defaults."""
        
        self.ticket_type = widgets.Dropdown(
            options=['Order', 'Trouble', 'Maintenance'],
            value='Order',
            description='Ticket Type:',
            style={'description_width': '150px'},
            layout=widgets.Layout(width='400px')
        )
        
        self.order_type = widgets.Dropdown(
            options=['install', 'repair', 'upgrade', 'maintenance'],
            value='install',
            description='Order Type:',
            style={'description_width': '150px'},
            layout=widgets.Layout(width='400px')
        )
        
        self.priority = widgets.Dropdown(
            options=['Critical', 'High', 'Medium', 'Low'],
            value='Critical',
            description='Priority:',
            style={'description_width': '150px'},
            layout=widgets.Layout(width='400px')
        )
        
        self.required_skill = widgets.Dropdown(
            options=[
                'Fiber ONT installation', 'Fiber splicing', 'Copper installation',
                'Router configuration', 'Video STB setup', 'Line repair',
                'Network troubleshooting', 'Equipment upgrade'
            ],
            value='Fiber ONT installation',
            description='Required Skill:',
            style={'description_width': '150px'},
            layout=widgets.Layout(width='400px')
        )
        
        self.duration_min = widgets.Dropdown(
            options=[30, 60, 90, 120, 150, 180],
            value=90,
            description='Duration (min):',
            style={'description_width': '150px'},
            layout=widgets.Layout(width='400px')
        )
        
        self.street = widgets.Text(
            value='123 Broadway',
            description='Street:',
            style={'description_width': '150px'},
            layout=widgets.Layout(width='400px')
        )
        
        self.city = widgets.Text(
            value='New York',
            description='City:',
            style={'description_width': '150px'},
            layout=widgets.Layout(width='400px')
        )
        
        self.state = widgets.Dropdown(
            options=['NY', 'CA', 'TX', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI'],
            value='NY',
            description='State:',
            style={'description_width': '150px'},
            layout=widgets.Layout(width='400px')
        )
        
        self.postal_code = widgets.Text(
            value='10007',
            description='Postal Code:',
            style={'description_width': '150px'},
            layout=widgets.Layout(width='400px')
        )
        
        self.customer_latitude = widgets.FloatText(
            value=40.712776,
            description='Latitude:',
            style={'description_width': '150px'},
            layout=widgets.Layout(width='400px')
        )
        
        self.customer_longitude = widgets.FloatText(
            value=-74.005974,
            description='Longitude:',
            style={'description_width': '150px'},
            layout=widgets.Layout(width='400px')
        )
        
        self.appointment_start = widgets.Text(
            value='2025-11-15T09:00',
            description='Start Time:',
            placeholder='YYYY-MM-DDTHH:MM',
            style={'description_width': '150px'},
            layout=widgets.Layout(width='400px')
        )
        
        self.appointment_end = widgets.Text(
            value='2025-11-15T10:30',
            description='End Time:',
            placeholder='YYYY-MM-DDTHH:MM',
            style={'description_width': '150px'},
            layout=widgets.Layout(width='400px')
        )
        
        self.submit_btn = widgets.Button(
            description='Submit Dispatch',
            button_style='info',
            layout=widgets.Layout(width='200px', height='40px')
        )
        self.submit_btn.on_click(self._on_submit)
        
        self.clear_btn = widgets.Button(
            description='Clear Form',
            button_style='warning',
            layout=widgets.Layout(width='200px', height='40px')
        )
        self.clear_btn.on_click(self._on_clear)
        
        # NEW: Add Get Insights button
        self.insights_btn = widgets.Button(
            description='📊 Get Strategic Insights',
            button_style='success',
            layout=widgets.Layout(width='250px', height='40px')
        )
        self.insights_btn.on_click(self._on_insights)
        
        self.output = widgets.Output()
        
    def _validate_form(self):
        """Validate form inputs."""
        errors = []
        
        if not self.street.value.strip():
            errors.append("Street is required")
        if not self.city.value.strip():
            errors.append("City is required")
        if not self.postal_code.value.strip():
            errors.append("Postal code is required")
        
        try:
            float(self.customer_latitude.value)
            float(self.customer_longitude.value)
        except:
            errors.append("Invalid latitude/longitude")
        
        try:
            datetime.strptime(self.appointment_start.value, '%Y-%m-%dT%H:%M')
            datetime.strptime(self.appointment_end.value, '%Y-%m-%dT%H:%M')
        except:
            errors.append("Invalid date/time format (use YYYY-MM-DDTHH:MM)")
        
        return errors
    
    def _on_submit(self, btn):
        """Handle form submission."""
        with self.output:
            self.output.clear_output()
            
            errors = self._validate_form()
            if errors:
                print("❌ Validation Errors:")
                for err in errors:
                    print(f"  • {err}")
                return
            
            print("🔄 Processing submission...\n")
            
            ticket_data = {
                'ticket_type': self.ticket_type.value,
                'order_type': self.order_type.value,
                'priority': self.priority.value,
                'required_skill': self.required_skill.value,
                'duration_min': self.duration_min.value,
                'street': self.street.value,
                'city': self.city.value,
                'state': self.state.value,
                'postal_code': self.postal_code.value,
                'customer_latitude': self.customer_latitude.value,
                'customer_longitude': self.customer_longitude.value,
                'appointment_start': self.appointment_start.value,
                'appointment_end': self.appointment_end.value
            }
            
            result = submit_dispatch_ticket(self.spark, ticket_data)
            
            self.output.clear_output()
            
            if result['success']:
                print("\n" + "="*80)
                print("✅ DISPATCH CREATED & OPTIMIZED SUCCESSFULLY")
                print("="*80)
                print(f"\n📋 Dispatch ID:           {result['dispatch_id']}")
                print(f"👤 Assigned Technician:   {result.get('technician_name', 'None')} ({result.get('technician_id', 'N/A')})")
                print(f"📍 Distance:              {result.get('distance_km', 0):.2f} km")
                print(f"📊 Productive Prob:       {result.get('p_productive', 0)*100:.1f}%")
                print(f"🔧 First-Time-Fix Prob:   {result.get('p_ftf', 0)*100:.1f}%")
                
                if result.get('all_constraints_met'):
                    print(f"✅ Constraints:           All Met")
                else:
                    print(f"⚠️  Constraints:           Partial")
                
                print(f"💡 Reason:                {result.get('reason', 'N/A')}")
                print("="*80)
                
                # Display explainable AI reasoning
                print("\n" + "="*80)
                print("🧠 AI ASSIGNMENT EXPLANATION")
                print("="*80)
                print(result.get('ai_explanation', 'No explanation available'))
                print("="*80)
            else:
                print("\n" + "="*80)
                print("❌ ERROR")
                print("="*80)
                print(f"{result.get('message', 'Unknown error')}")
                print("="*80)
    
    def _on_clear(self, btn):
        """Clear all form fields."""
        self.street.value = ''
        self.city.value = ''
        self.postal_code.value = ''
        self.customer_latitude.value = 0.0
        self.customer_longitude.value = 0.0
        self.appointment_start.value = ''
        self.appointment_end.value = ''
        self.output.clear_output()
    
    def _on_insights(self, btn):
        """NEW: Handle strategic insights request."""
        with self.output:
            self.output.clear_output()
            
            print("🔄 Generating strategic insights...\n")
            print("This may take a minute as we analyze all optimized dispatches...\n")
            
            try:
                # Initialize and run the strategy insights agent
                strategy_agent = StrategyInsightsAgent(
                    spark=self.spark,
                    catalog="hackathon",
                    schema="hackathon_fiber_vault"
                )
                
                # Generate the memo
                memo = strategy_agent.analyze(save_to_file=False)
                
                # Display is already done by the analyze() method
                # which prints the memo to console
                
            except Exception as e:
                import traceback
                self.output.clear_output()
                print("\n" + "="*80)
                print("❌ ERROR GENERATING INSIGHTS")
                print("="*80)
                print(f"{str(e)}")
                print("\n" + "="*80)
                traceback.print_exc()
    
    def display(self):
        """Display the complete UI."""
        
        # Custom CSS
        style_html = HTML("""
        <style>
        .widget-label { color: #00d4ff !important; font-weight: bold; }
        .widget-dropdown select, .widget-text input, .widget-float input {
            background-color: #2d2d2d !important;
            color: #ffffff !important;
            border: 1px solid #00d4ff !important;
        }
        </style>
        """)
        
        # Header
        header = HTML("""
        <div style='background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%); 
                    padding: 30px; border-radius: 10px; margin-bottom: 20px;
                    border: 2px solid #00d4ff;'>
            <h1 style='color: #00d4ff; margin: 0; font-size: 28px;'>
                🎯 SmartDispatch AI Ticketing System
            </h1>
            <p style='color: #ffffff; margin: 10px 0 0 0; font-size: 14px;'>
                Submit new dispatch requests with automatic AI-powered technician assignment
            </p>
        </div>
        """)
        
        # Form sections
        section1 = widgets.VBox([
            widgets.HTML("<h3 style='color: #00d4ff;'>📋 Ticket Details</h3>"),
            self.ticket_type,
            self.order_type,
            self.priority,
            self.required_skill,
            self.duration_min
        ])
        
        section2 = widgets.VBox([
            widgets.HTML("<h3 style='color: #00d4ff;'>📍 Location Details</h3>"),
            self.street,
            self.city,
            self.state,
            self.postal_code,
            self.customer_latitude,
            self.customer_longitude
        ])
        
        section3 = widgets.VBox([
            widgets.HTML("<h3 style='color: #00d4ff;'>📅 Appointment Time</h3>"),
            self.appointment_start,
            self.appointment_end
        ])
        
        # NEW: Updated button box with insights button
        button_box = widgets.HBox(
            [self.submit_btn, self.clear_btn, self.insights_btn], 
            layout=widgets.Layout(margin='20px 0', justify_content='flex-start')
        )
        
        # Display everything
        display(style_html)
        display(header)
        display(section1)
        display(section2)
        display(section3)
        display(button_box)
        display(self.output)

# ============================================================
# USAGE
# ============================================================
ui = DispatchTicketingUI(spark)
ui.display()