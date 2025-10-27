def adjust_for_wind(base_rate, wind_speed, wind_direction):
    multiplier = 1 + 0.05 * (wind_speed ** 0.5)
    return base_rate * multiplier
