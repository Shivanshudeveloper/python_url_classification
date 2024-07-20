from datetime import datetime
import pytz

def convert_to_z_format(db_timestamp):
    # Parse the database timestamp
    dt = datetime.strptime(db_timestamp, '%Y-%m-%d %H:%M:%S')
    
    # Convert to UTC time
    utc_time = pytz.utc.localize(dt)
    
    # Convert to 'Z' format
    z_format_time = utc_time.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    
    return z_format_time

def convert_to_local_time(z_format_time, timezone_str='Asia/Kolkata'):
    # Parse the 'Z' format time
    utc_time = datetime.strptime(z_format_time, '%Y-%m-%dT%H:%M:%S.%fZ')
    utc_time = pytz.utc.localize(utc_time)
    
    # Convert to the local time zone
    local_tz = pytz.timezone(timezone_str)
    local_time = utc_time.astimezone(local_tz)
    
    # Format the local time
    local_time_str = local_time.strftime('%Y-%m-%d %H:%M:%S')
    
    return local_time_str

# Example usage
db_timestamp = '2024-07-18 15:38:47'
z_format_time = convert_to_z_format(db_timestamp)
print(f"Z format time: {z_format_time}")

local_time = convert_to_local_time(z_format_time)
print(f"Local time (India): {local_time}")
