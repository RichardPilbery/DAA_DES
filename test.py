import simpy
import random
from collections import defaultdict

# Hourly activity pattern (proportions summing to 1)
hourly_activity_pattern = [
    0.02, 0.01, 0.01, 0.01, 0.01, 0.02,  # Midnight to 6 AM
    0.04, 0.05, 0.08, 0.10, 0.12, 0.13,  # 6 AM to Noon
    0.15, 0.12, 0.10, 0.08, 0.05, 0.04,  # Noon to 6 PM
    0.03, 0.02, 0.02, 0.01, 0.01, 0.01   # 6 PM to Midnight
]

def generate_daily_calls():
    # Randomly pick total calls for the day (e.g., 2 to 10)
    total_calls = random.randint(2, 10)
    
    # Scale hourly activity pattern
    hourly_calls = [p * total_calls for p in hourly_activity_pattern]
    
    # Probabilistically round to integers
    rounded_hourly_calls = [
        int(c) + (1 if random.random() < (c - int(c)) else 0) for c in hourly_calls
    ]
    
    return rounded_hourly_calls

def call_generator(env, daily_call_schedule):
    for hour, num_calls in enumerate(daily_call_schedule):
        print(f"hour {hour} and number of calls {num_calls}")
        if num_calls > 0:
            inter_arrival_times = [
                random.expovariate(num_calls / 3600) for _ in range(num_calls)
            ]
            for inter_arrival in sorted(inter_arrival_times):
                print(inter_arrival)
                if(inter_arrival > 0):
                    yield env.timeout(((hour * 3600) + inter_arrival) - env.now)
                    print(f"Call at {env.now:.2f} seconds")

# SimPy environment
env = simpy.Environment()

# Generate daily call schedule
daily_call_schedule = generate_daily_calls()
print(daily_call_schedule)
print("Daily Call Schedule:", daily_call_schedule)

# Start call generator
env.process(call_generator(env, daily_call_schedule))
env.run(until=86400)  # Simulate 1 day
