from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os
import logging
import json
import pytz
import requests
import time
from collections import defaultdict
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Add cache for OpenAI responses
CATEGORY_CACHE = {}

# Database configuration
try:
    engine = create_engine(f'mysql+mysqlconnector://{os.getenv("DB_USERNAME")}:{os.getenv("DB_PASSWORD")}@{os.getenv("DB_HOST")}/{os.getenv("DB_NAME")}')
    Session = sessionmaker(bind=engine)
except Exception as e:
    logging.error(f"Error setting up database connection: {e}")

# Azure OpenAI configuration
OPENAI_ENDPOINT = "https://b2brocket-openai2.openai.azure.com/openai/deployments/gpt-35-turbo/chat/completions"
API_VERSION = "2024-02-15-preview"
API_KEY = os.getenv("AZURE_OPENAI_KEY")

# Hardcoded productivity policy for development companies
HARDCODED_POLICY = {
    # Core Productive
    "Visual Studio Code": "Core Productive",
    "VSCode": "Core Productive",
    "IntelliJ IDEA": "Core Productive",
    "PyCharm": "Core Productive",
    "Git": "Core Productive",
    "GitHub": "Core Productive",
    "GitLab": "Core Productive",
    "Docker": "Core Productive",
    "Postman": "Core Productive",
    "Jira": "Core Productive",
    "Confluence": "Core Productive",
    
    # Productive
    "Microsoft Teams": "Productive",
    "Slack": "Productive",
    "Zoom": "Productive",
    "Google Meet": "Productive",
    "Gmail": "Productive",
    
    # Unproductive
    "YouTube": "Unproductive",
    "Netflix": "Unproductive",
    "Spotify": "Unproductive",
    "Facebook": "Unproductive",
    "Instagram": "Unproductive",
    
    # Idle
    "System Settings": "Idle",
    "File Explorer": "Idle",
    
    # Away
    "Lock Screen": "Away",
    "Screensaver": "Away"
}

def query_openai(title, app_name):
    """Query OpenAI API for productivity classification"""
    try:
        system_prompt = """
        Classify user activity into exactly one category: 
        [Core Productive, Productive, Idle, Unproductive, Away].
        Company Type: Software Development
        Guidelines:
        - Core Productive: Direct coding/development tasks
        - Productive: Meetings, communications, documentation
        - Idle: System/file management
        - Unproductive: Entertainment, social media
        - Away: No user activity
        """

        user_prompt = f"""
        Application: {app_name}
        Title/Activity: {title}

        Examples:
        - "config.py" in VSCode → Core Productive
        - "Team Standup" in Zoom → Productive
        - "Music Playlist" in Spotify → Unproductive
        - "Settings" in System → Idle
        - "Lock Screen" → Away
        """

        response = requests.post(
            f"{OPENAI_ENDPOINT}?api-version={API_VERSION}",
            headers={"Content-Type": "application/json", "api-key": API_KEY},
            json={
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "max_tokens": 800,
                "temperature": 0.1,
                "top_p": 0.95
            }
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        logging.error(f"OpenAI API error: {e}")
        return "Idle"


# Add retry logic for OpenAI API
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def query_openai_with_retry(title, app_name):
    return query_openai(title, app_name)

def predict_category(page_title, app_name):
    try:
        # Clean inputs
        app_name = app_name.strip() or "Unknown"
        page_title = page_title.strip() or "No Title"
        
        # Check cache first
        cache_key = f"{app_name}|{page_title}"
        if cache_key in CATEGORY_CACHE:
            return CATEGORY_CACHE[cache_key], 1.0, False
        
        # Check hardcoded policy
        if app_name in HARDCODED_POLICY:
            result = HARDCODED_POLICY[app_name]
            CATEGORY_CACHE[cache_key] = result
            return result, 1.0, False
        
        # Query OpenAI with rate limiting
        time.sleep(0.2)  # Add delay between requests
        category = query_openai_with_retry(page_title, app_name)
        CATEGORY_CACHE[cache_key] = category
        return category, 1.0, False
        
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return "Idle", 0.0, False

def get_user_activities(device_uid, date_str):
    session = Session()
    try:
        logging.info(f"Fetching activities for {device_uid} on {date_str}")
        
        # Convert date string to datetime object
        target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        
        result = session.execute(text("""
            SELECT page_title, app_name, timestamp 
            FROM user_activity 
            WHERE DATE(timestamp) = :date 
            AND user_uid = :device_uid 
            ORDER BY timestamp
        """), {
            "date": target_date,
            "device_uid": device_uid
        }).fetchall()
        
        # Convert timestamps to local timezone
        activities = []
        local_tz = pytz.timezone('Asia/Kolkata')
        
        for row in result:
            db_timestamp = row[2]
            # Convert naive datetime to aware UTC first
            utc_time = pytz.utc.localize(db_timestamp)
            # Convert to local timezone
            local_time = utc_time.astimezone(local_tz)
            
            activities.append((
                row[0],  # page_title
                row[1],  # app_name
                local_time.replace(tzinfo=None)  # Remove tzinfo for compatibility
            ))
        
        logging.info(f"Found {len(activities)} activities")
        return activities
        
    except Exception as e:
        logging.error(f"Error fetching activities: {e}")
        return []
    finally:
        session.close()


def calculate_productivity_internal(activities):
    try:
        if not activities:
            return [
                [
                    {"productivity": "away", "percent": "100.00%"},
                    {"productivity": "", "percent": ""},
                    {"productivity": "", "percent": ""},
                    {"productivity": "", "percent": ""},
                    {"productivity": "", "percent": ""},
                    {"productivity": "", "percent": ""}
                ] for _ in range(12)  # 9AM to 9PM = 12 hours
            ]

        # Get the date from first activity
        first_activity_time = activities[0][2]
        target_date = first_activity_time.date()
        
        # Initialize time slots (9AM to 9PM)
        hours = [
            datetime(target_date.year, target_date.month, target_date.day, h, 0)
            for h in range(9, 21)
        ]
        
        # Track time spent in each category per hour
        hourly_times = {hour: defaultdict(float) for hour in hours}
        
        # Sort activities by time
        activities.sort(key=lambda x: x[2])
        prev_time = None
        
        for i, (title, app, curr_time) in enumerate(activities):
            # Calculate duration
            if prev_time:
                duration = (curr_time - prev_time).total_seconds() / 60  # minutes
            else:
                duration = 1  # Assume 1 minute for first activity
            
            # Get category
            category, _, _ = predict_category(title, app)
            
            # Distribute duration across overlapping hours
            start = prev_time or curr_time
            end = curr_time
            current = start.replace(second=0, microsecond=0)
            
            while current < end:
                hour_key = current.replace(minute=0)
                if hour_key in hourly_times:
                    mins = min((end - current).total_seconds() / 60, 60 - current.minute)
                    hourly_times[hour_key][category.lower()] += mins
                current += timedelta(minutes=60 - current.minute)
            
            prev_time = curr_time
        
        # Calculate percentages
        productivity_data = []
        for hour in hours:
            total_mins = sum(hourly_times[hour].values()) or 1
            away_mins = 60 - total_mins
            
            percentages = {
                'core productive': (hourly_times[hour].get('core productive', 0)) / 60 * 100,
                'productive': (hourly_times[hour].get('productive', 0)) / 60 * 100,
                'idle': (hourly_times[hour].get('idle', 0)) / 60 * 100,
                'unproductive': (hourly_times[hour].get('unproductive', 0)) / 60 * 100,
                'away': (away_mins / 60) * 100
                }
            
            productivity_data.append([
                {"productivity": "core productive", "percent": f"{percentages['core productive']:.2f}%"},
                {"productivity": "productive", "percent": f"{percentages['productive']:.2f}%"},
                {"productivity": "idle", "percent": f"{percentages['idle']:.2f}%"},
                {"productivity": "unproductive", "percent": f"{percentages['unproductive']:.2f}%"},
                {"productivity": "away", "percent": f"{percentages['away']:.2f}%"},
                {"productivity": "", "percent": ""}
                ])
        
        return productivity_data

    except Exception as e:
        logging.error(f"Error calculating productivity: {e}")
        return []

def get_all_device_ids(organization_uid):
    """Fetch actual devices from database"""
    logging.info(f"Fetching devices for organization: {organization_uid}")
    session = Session()
    try:
        result = session.execute(text("""
            SELECT device_uid, user_name 
            FROM devices 
            WHERE organization_uid = :organization_uid
        """), {"organization_uid": organization_uid}).fetchall()
        
        return [{'device_uid': row[0], 'user_name': row[1]} for row in result]
    except Exception as e:
        logging.error(f"Error fetching devices: {e}")
        return []
    finally:
        session.close()

def calculate_working_hours(activities):
    """Calculate actual working hours from activities"""
    try:
        if not activities:
            return "0h 0m"
            
        first_activity = activities[0][2]  # Tuple: (page_title, app_name, timestamp)
        last_activity = activities[-1][2]
        working_duration = last_activity - first_activity
        
        hours, remainder = divmod(working_duration.total_seconds(), 3600)
        minutes, _ = divmod(remainder, 60)
        return f"{int(hours)}h {int(minutes)}m"
    except Exception as e:
        logging.error(f"Error calculating working hours: {e}")
        return "0h 0m"

@app.route('/calculate_hourly_productivity', methods=['GET'])
def calculate_hourly_productivity():
    date_str = request.args.get('date')
    organization_uid = request.args.get('organization_uid')
    
     # Set up timezone (adjust to your local timezone)
    local_tz = pytz.timezone('Asia/Kolkata')
    print(f"date {date_str}")
    try:
        if date_str:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
        else:
            # Get current date in specified timezone
            date_obj = datetime.now(local_tz).date()
    except ValueError:
        return jsonify({"error": "Invalid date format, use YYYY-MM-DD"}), 400

    if not organization_uid:
        return jsonify({"error": "organization_uid is required"}), 400

    try:
        # Get real devices from database
        date_for_db = date_obj.strftime('%Y-%m-%d')
        logging.info(f"Processing date: {date_for_db} for organization: {organization_uid}")

        devices = get_all_device_ids(organization_uid)
        
        if not devices:
            return jsonify({"error": "No devices found for this organization"}), 404

        all_productivity_data = []

        for device in devices:
            print(device)
            activities = get_user_activities(device['device_uid'], date_for_db)
            print(f"activites: {len(activities)}")
            all_productivity_data.append({
                'name': device['user_name'],
                'workingHour': calculate_working_hours(activities),
                'productivityRecord': calculate_productivity_internal(activities)
            })
            
        return jsonify(all_productivity_data)

    except Exception as e:
        logging.error(f"Error in hourly productivity calculation: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/predict_productivity', methods=['POST'])
def predict_productivity_route():
    data = request.get_json()
    title = data.get('title', '')
    app_name = data.get('app_name', 'Unknown')

    category, confidence, _ = predict_category(title, app_name)
    return jsonify({
        "category": category,
        "productivity": category.lower(),
        "confidence": f"{confidence * 100:.2f}%"
    })

if __name__ == '__main__':
    app.run(debug=True)