CATEGORICAL_COLUMNS = [
    'restaurant_id',
    'country_id',
    'day_of_week_order',
    'order_hour',
    'business_type',
]
NUMERICAL_COLUMNS = [
    # 'user_orders_30_days_same_partner',
    # 'total_amount_euros',

    # 'shipping_amount_euros',
    # 'commission',
    # 'online_payment',
    # 'total_orders_previous_hour_past_1hour_order',
    # 'PFR_orders_previous_hour_past_1hour_order',
    # 'total_orders_last_7days',
    # 'PFR_orders_last_7days',
    # 'total_orders_last_day',
    # 'PFR_orders_last_day',
    # 'total_orders_7days_before',
    # 'PFR_orders_7days_before',
    # 'total_orders_7days_before_same_hour',
    # 'PFR_orders_7days_before_same_hour',
    'promised_delivery_time_min',
    # 'estimated_prep_time_min',
    # 'estimated_prep_buffer',
    # 'estimated_driving_time_min',
    # 'estimated_courier_delay_min',
    # 'holiday',
    # 'holiday_type',
    'minutes_to_close'
]
TARGET = 'partner_confirm_order'

COLUMNS = CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS
