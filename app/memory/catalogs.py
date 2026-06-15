"""Approved v1 memory key catalogs and expiration defaults."""

from datetime import datetime, timedelta
import re


PROFILE_FIELD_KEYS = {
    "display_name",
    "timezone",
    "preferred_language",
    "preferred_units",
    "birth_year",
    "age",
    "sex",
    "gender_identity",
    "height_cm",
    "current_bodyweight_kg",
    "target_bodyweight_kg",
    "body_fat_percent",
    "long_term_goal",
    "primary_training_goal",
    "secondary_training_goal",
    "target_event",
    "target_event_date",
    "training_age_years",
    "experience_level",
    "primary_sport",
    "secondary_sports",
    "preferred_training_style",
    "preferred_training_days_per_week",
    "available_training_days",
    "typical_session_duration_minutes",
    "weekly_schedule_constraints",
    "training_location",
    "available_equipment",
    "home_gym_equipment",
    "gym_access",
    "current_program_name",
    "current_program_start_date",
    "current_program_goal",
    "bench_press_1rm_kg",
    "squat_1rm_kg",
    "deadlift_1rm_kg",
    "overhead_press_1rm_kg",
    "recent_best_lifts_json",
    "cardio_baseline",
    "mobility_limitations",
    "long_term_injury_history",
    "medical_constraints_user_reported",
    "exercise_preferences",
    "exercise_dislikes",
    "food_preferences",
    "food_dislikes",
    "dietary_restrictions",
    "allergies_user_reported",
    "nutrition_goal",
    "supplement_preferences",
}


RECENT_STATUS_KEYS = {
    "sleep_quality_last_night",
    "sleep_duration_hours",
    "sleep_consistency",
    "morning_readiness",
    "energy_level",
    "motivation_level",
    "stress_level",
    "mental_fatigue",
    "physical_fatigue",
    "muscle_soreness_general",
    "muscle_soreness_body_parts",
    "joint_pain",
    "injury_flare_up",
    "illness_symptoms",
    "medication_change_user_reported",
    "recovery_score_user_reported",
    "resting_heart_rate_user_reported",
    "bodyweight_today_kg",
    "bodyweight_trend_recent",
    "appetite_level",
    "digestion_status",
    "hydration_status",
    "alcohol_intake_recent",
    "travel_status",
    "jet_lag_status",
    "work_schedule_pressure",
    "available_training_time_today",
    "available_training_days_this_week",
    "missed_recent_sessions",
    "completed_recent_sessions",
    "last_session_rpe",
    "last_session_notes",
    "current_session_readiness",
    "training_environment_today",
    "equipment_availability_today",
    "pain_during_last_session",
    "exercise_to_avoid_temporarily",
    "user_requested_plan_change",
    "plan_adherence_recent",
    "nutrition_adherence_recent",
    "protein_intake_recent",
    "calorie_intake_recent",
    "meal_timing_issue",
    "menstrual_cycle_phase_user_reported",
    "mood_state",
    "confidence_level",
    "time_constraint_reason",
    "recovery_action_taken",
    "coach_follow_up_needed",
    "safety_attention_needed",
}


DAILY_STATUS_KEYS = {
    "sleep_quality_last_night",
    "sleep_duration_hours",
    "sleep_consistency",
    "morning_readiness",
    "energy_level",
    "motivation_level",
    "appetite_level",
    "digestion_status",
    "hydration_status",
    "mood_state",
    "available_training_time_today",
    "current_session_readiness",
    "training_environment_today",
    "equipment_availability_today",
}

THREE_DAY_STATUS_KEYS = {
    "stress_level",
    "mental_fatigue",
    "physical_fatigue",
    "muscle_soreness_general",
    "muscle_soreness_body_parts",
    "travel_status",
    "jet_lag_status",
    "work_schedule_pressure",
    "last_session_rpe",
    "last_session_notes",
}

SEVEN_DAY_STATUS_KEYS = {
    "joint_pain",
    "injury_flare_up",
    "illness_symptoms",
    "medication_change_user_reported",
    "pain_during_last_session",
    "exercise_to_avoid_temporarily",
    "coach_follow_up_needed",
    "safety_attention_needed",
}

FOURTEEN_DAY_STATUS_KEYS = {
    "available_training_days_this_week",
    "plan_adherence_recent",
    "nutrition_adherence_recent",
    "bodyweight_trend_recent",
    "missed_recent_sessions",
    "completed_recent_sessions",
    "user_requested_plan_change",
    "protein_intake_recent",
    "calorie_intake_recent",
    "meal_timing_issue",
    "recovery_score_user_reported",
    "resting_heart_rate_user_reported",
    "bodyweight_today_kg",
    "time_constraint_reason",
    "recovery_action_taken",
}


def normalize_key(value: str) -> str:
    """Normalize user-entered memory keys into stable snake_case strings."""

    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", value.strip().lower())
    return re.sub(r"_+", "_", cleaned).strip("_")


def default_valid_until(status_key: str, observed_at: datetime) -> datetime:
    """Return the v1 default expiration timestamp for one recent-status key."""

    key = normalize_key(status_key)
    if key in DAILY_STATUS_KEYS:
        return observed_at + timedelta(days=1)
    if key in THREE_DAY_STATUS_KEYS:
        return observed_at + timedelta(days=3)
    if key in SEVEN_DAY_STATUS_KEYS:
        return observed_at + timedelta(days=7)
    if key in FOURTEEN_DAY_STATUS_KEYS:
        return observed_at + timedelta(days=14)
    return observed_at + timedelta(days=3)

