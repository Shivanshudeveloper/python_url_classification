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
from threading import RLock
import hashlib

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Enhanced cache for OpenAI responses with thread safety
CATEGORY_CACHE = {}
CACHE_LOCK = RLock()
CACHE_SIZE_LIMIT = 10000

# Statistics tracking
STATS = {
    'openai_calls': 0,
    'openai_failures': 0,
    'cache_hits': 0,
    'fallback_used': 0,
    'hardcoded_hits': 0
}

# Database configuration for PostgreSQL
try:
    DATABASE_URL = f"postgresql://{os.getenv('DB_USERNAME')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT', '5432')}/{os.getenv('DB_NAME')}"
    
    engine = create_engine(
        DATABASE_URL,
        pool_size=20,
        max_overflow=30,
        pool_pre_ping=True,
        pool_recycle=3600,
        echo=False
    )
    Session = sessionmaker(bind=engine)
    logging.info("PostgreSQL database connection established successfully")
except Exception as e:
    logging.error(f"Error setting up PostgreSQL database connection: {e}")
    raise

# Azure OpenAI configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://b2brocket-openai2.openai.azure.com")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-35-turbo")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
API_KEY = os.getenv("AZURE_OPENAI_KEY")

OPENAI_ENDPOINT = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/chat/completions"

if not API_KEY:
    logging.warning("AZURE_OPENAI_KEY not found in environment variables")
    
logging.info(f"OpenAI Endpoint configured: {OPENAI_ENDPOINT}")

# Enhanced productivity policy
HARDCODED_POLICY = {
    # Core Productive
    "Visual Studio Code": "Core Productive",
    "VSCode": "Core Productive",
    "IntelliJ IDEA": "Core Productive",
    "PyCharm": "Core Productive",
    "WebStorm": "Core Productive",
    "Sublime Text": "Core Productive",
    "Atom": "Core Productive",
    "Eclipse": "Core Productive",
    "Git": "Core Productive",
    "GitHub": "Core Productive",
    "GitLab": "Core Productive",
    "Bitbucket": "Core Productive",
    "Docker": "Core Productive",
    "Kubernetes": "Core Productive",
    "Postman": "Core Productive",
    "Insomnia": "Core Productive",
    "Jira": "Core Productive",
    "Confluence": "Core Productive",
    "Azure DevOps": "Core Productive",
    "Jenkins": "Core Productive",
    "Terminal": "Core Productive",
    "Command Prompt": "Core Productive",
    "PowerShell": "Core Productive",
    "MySQL Workbench": "Core Productive",
    "pgAdmin": "Core Productive",
    "MongoDB Compass": "Core Productive",
    "Redis Desktop Manager": "Core Productive",
    
    # Productive
    "Microsoft Teams": "Productive",
    "Slack": "Productive",
    "Discord": "Productive",
    "Zoom": "Productive",
    "Google Meet": "Productive",
    "Microsoft Outlook": "Productive",
    "Gmail": "Productive",
    "Thunderbird": "Productive",
    "Notion": "Productive",
    "Trello": "Productive",
    "Asana": "Productive",
    "Monday.com": "Productive",
    "Google Docs": "Productive",
    "Microsoft Word": "Productive",
    "Google Sheets": "Productive",
    "Microsoft Excel": "Productive",
    "Google Drive": "Productive",
    "OneDrive": "Productive",
    "Dropbox": "Productive",
    "Stack Overflow": "Productive",
    "GitHub Pages": "Productive",
    "Documentation": "Productive",
    
    # Unproductive
    "YouTube": "Unproductive",
    "Netflix": "Unproductive",
    "Prime Video": "Unproductive",
    "Disney+": "Unproductive",
    "Hulu": "Unproductive",
    "Spotify": "Unproductive",
    "Apple Music": "Unproductive",
    "Facebook": "Unproductive",
    "Instagram": "Unproductive",
    "Twitter": "Unproductive",
    "LinkedIn": "Unproductive",
    "TikTok": "Unproductive",
    "Reddit": "Unproductive",
    "WhatsApp": "Unproductive",
    "Telegram": "Unproductive",
    "Steam": "Unproductive",
    "Epic Games": "Unproductive",
    "Gaming": "Unproductive",
    
    # Idle
    "System Settings": "Idle",
    "Control Panel": "Idle",
    "File Explorer": "Idle",
    "Finder": "Idle",
    "Task Manager": "Idle",
    "Activity Monitor": "Idle",
    "System Preferences": "Idle",
    "Windows Update": "Idle",
    "Software Update": "Idle",
    
    # Away
    "Lock Screen": "Away",
    "Screensaver": "Away",
    "Sleep": "Away",
    "Hibernate": "Away"
}

def get_cache_key(title, app_name):
    """Generate a consistent cache key"""
    content = f"{app_name.lower().strip()}|{title.lower().strip()}"
    return hashlib.md5(content.encode()).hexdigest()

def manage_cache_size():
    """Remove oldest entries if cache exceeds size limit"""
    with CACHE_LOCK:
        if len(CATEGORY_CACHE) > CACHE_SIZE_LIMIT:
            items_to_remove = len(CATEGORY_CACHE) // 5
            keys_to_remove = list(CATEGORY_CACHE.keys())[:items_to_remove]
            for key in keys_to_remove:
                del CATEGORY_CACHE[key]

def classify_with_fallback(title, app_name):
    """FAST fallback classification - NO EXTERNAL CALLS"""
    title_lower = title.lower()
    app_lower = app_name.lower()
    
    logging.info(f"Using FAST fallback classification for: {app_name} - {title}")
    
    # Development tools and IDEs
    dev_tools = ['code', 'studio', 'intellij', 'pycharm', 'webstorm', 'sublime', 'atom', 'eclipse', 'vim', 'emacs']
    if any(tool in app_lower for tool in dev_tools):
        logging.info(f"FAST: Classified as Core Productive due to dev tool: {app_name}")
        return "Core Productive"
    
    # Code file extensions
    code_extensions = ['.py', '.js', '.html', '.css', '.sql', '.json', '.xml', '.md', '.java', '.cpp', '.c', '.php', '.rb', '.go', '.ts', '.jsx', '.tsx', '.vue']
    if any(ext in title_lower for ext in code_extensions):
        logging.info(f"FAST: Classified as Core Productive due to file extension in: {title}")
        return "Core Productive"
    
    # Version control
    version_control = ['git', 'github', 'gitlab', 'bitbucket', 'svn']
    if any(vc in app_lower for vc in version_control):
        return "Core Productive"
    
    # Communication and meetings
    comm_apps = ['teams', 'slack', 'zoom', 'meet', 'discord', 'skype']
    meeting_keywords = ['meeting', 'call', 'standup', 'review', 'presentation', 'demo', 'sync', 'discussion']
    if any(comm in app_lower for comm in comm_apps) or any(keyword in title_lower for keyword in meeting_keywords):
        return "Productive"
    
    # Documentation and productivity
    doc_apps = ['notion', 'confluence', 'docs', 'word', 'excel', 'sheets', 'trello', 'asana', 'jira']
    if any(doc in app_lower for doc in doc_apps):
        return "Productive"
    
    # Entertainment
    entertainment = ['youtube', 'netflix', 'spotify', 'facebook', 'instagram', 'twitter', 'tiktok', 'reddit', 'gaming', 'steam', 'epic']
    if any(ent in app_lower for ent in entertainment):
        return "Unproductive"
    
    # System and file management
    system_apps = ['explorer', 'finder', 'settings', 'control panel', 'task manager', 'activity monitor']
    if any(sys in app_lower for sys in system_apps):
        return "Idle"
    
    # Away indicators
    away_indicators = ['lock', 'screensaver', 'sleep', 'hibernate']
    if any(away in title_lower or away in app_lower for away in away_indicators):
        return "Away"
    
    # Default fallback
    logging.info(f"FAST: Using default Idle classification for: {app_name} - {title}")
    return "Idle"

def predict_category_fast(page_title, app_name):
    """FAST prediction with NO hanging - guaranteed to return quickly"""
    try:
        # Clean and normalize inputs
        app_name = app_name.strip() if app_name else "Unknown"
        page_title = page_title.strip() if page_title else "No Title"
        
        logging.info(f"FAST prediction for: {app_name} - {page_title}")
        
        # Generate cache key
        cache_key = get_cache_key(page_title, app_name)
        print(cache_key)
        # Check cache first (thread-safe)
        with CACHE_LOCK:
            if cache_key in CATEGORY_CACHE:
                STATS['cache_hits'] += 1
                logging.info(f"FAST: Cache hit for: {app_name} - {page_title}")
                return CATEGORY_CACHE[cache_key], 1.0, True
        
        # Check hardcoded policy (case-insensitive)
        app_name_lower = app_name.lower()
        for policy_app, category in HARDCODED_POLICY.items():
            if policy_app.lower() in app_name_lower or app_name_lower in policy_app.lower():
                STATS['hardcoded_hits'] += 1
                logging.info(f"FAST: Hardcoded policy hit: {app_name} -> {category}")
                with CACHE_LOCK:
                    CATEGORY_CACHE[cache_key] = category
                    print(CATEGORY_CACHE)
                    manage_cache_size()
                return category, 1.0, False
        print(" I am done wiht the cache part.")
        # Check for common patterns in title
        title_lower = page_title.lower()
        print(f"title_lower {title_lower}")
        # Code file extensions
        code_extensions = ['.py', '.js', '.html', '.css', '.sql', '.json', '.xml', '.md', '.java', '.cpp', '.c', '.php', '.rb', '.go', '.ts', '.jsx', '.tsx', '.vue', '.scss', '.sass', '.less']
        if any(ext in title_lower for ext in code_extensions):
            category = "Core Productive"
            logging.info(f"FAST: Pattern match (code extension): {page_title} -> {category}")
            with CACHE_LOCK:
                CATEGORY_CACHE[cache_key] = category
                manage_cache_size()
            return category, 1.0, False
        
        # Meeting keywords
        meeting_keywords = ['meeting', 'call', 'standup', 'review', 'presentation', 'demo', 'sync', 'discussion']
        if any(keyword in title_lower for keyword in meeting_keywords):
            category = "Productive"
            logging.info(f"FAST: Pattern match (meeting keyword): {page_title} -> {category}")
            with CACHE_LOCK:
                CATEGORY_CACHE[cache_key] = category
                manage_cache_size()
            return category, 1.0, False
        
        # Use FAST fallback - NO OpenAI calls for now to ensure response
        STATS['fallback_used'] += 1
        category = classify_with_fallback(page_title, app_name)
        
        # Cache the result
        with CACHE_LOCK:
            CATEGORY_CACHE[cache_key] = category
            manage_cache_size()
        
        logging.info(f"FAST: Final classification: {app_name} - {page_title} -> {category}")
        return category, 1.0, False
        
    except Exception as e:
        logging.error(f"FAST: Prediction error for {app_name}:{page_title} - {e}")
        # Always return a safe fallback
        return "Idle", 0.0, False

@app.route('/predict_productivity', methods=['POST'])
def predict_productivity_route():
    """GUARANTEED WORKING productivity prediction endpoint"""
    start_time = time.time()
    
    try:
        logging.info("=== PREDICT PRODUCTIVITY REQUEST START ===")
        
        data = request.get_json()
        
        if not data:
            logging.error("No JSON data provided")
            return jsonify({
                "error": "No JSON data provided",
                "category": "Idle",
                "productivity": "idle",
                "confidence": "0.00%",
                "cached": False,
                "status": "error"
            }), 400
            
        title = data.get('title', '')
        app_name = data.get('app_name', 'Unknown')
        
        logging.info(f"Processing: app='{app_name}', title='{title}'")

        # Use FAST prediction - guaranteed to work
        category, confidence, is_cached = predict_category_fast(title, app_name)
        
        result = {
            "category": category,
            "productivity": category.lower(),
            "confidence": f"{confidence * 100:.2f}%",
            "cached": is_cached,
            "status": "success",
            "processing_time_ms": round((time.time() - start_time) * 1000, 2)
        }
        
        logging.info(f"SUCCESS: {result}")
        logging.info("=== PREDICT PRODUCTIVITY REQUEST END ===")
        
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"CRITICAL ERROR in prediction endpoint: {e}")
        return jsonify({
            "error": "Internal server error",
            "details": str(e),
            "category": "Idle",
            "productivity": "idle",
            "confidence": "0.00%",
            "cached": False,
            "status": "error",
            "processing_time_ms": round((time.time() - start_time) * 1000, 2)
        }), 500

# Simple test endpoint
@app.route('/test', methods=['GET'])
def test_endpoint():
    """Simple test endpoint"""
    return jsonify({
        "status": "working",
        "message": "Flask app is running",
        "timestamp": datetime.now().isoformat()
    })

# Statistics endpoint
@app.route('/stats', methods=['GET'])
def get_stats():
    """Get API usage statistics"""
    total_predictions = sum(STATS.values())
    return jsonify({
        "total_predictions": total_predictions,
        "openai_calls": STATS['openai_calls'],
        "openai_failures": STATS['openai_failures'],
        "cache_hits": STATS['cache_hits'],
        "hardcoded_hits": STATS['hardcoded_hits'],
        "fallback_used": STATS['fallback_used'],
        "cache_size": len(CATEGORY_CACHE)
    })

# Keep all your other endpoints from the original code
def get_user_activities(device_uid, date_str):
    """Get user activities with PostgreSQL optimizations"""
    session = Session()
    try:
        logging.info(f"Fetching activities for {device_uid} on {date_str}")
        
        target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        
        result = session.execute(text("""
            SELECT page_title, app_name, timestamp 
            FROM user_activity 
            WHERE DATE(timestamp) = :date 
            AND user_uid = :device_uid 
            ORDER BY timestamp ASC
        """), {
            "date": target_date,
            "device_uid": device_uid
        }).fetchall()
        
        activities = []
        local_tz = pytz.timezone('Asia/Kolkata')
        
        for row in result:
            db_timestamp = row[2]
            
            if db_timestamp.tzinfo is None:
                utc_time = pytz.utc.localize(db_timestamp)
            else:
                utc_time = db_timestamp.astimezone(pytz.utc)
            
            local_time = utc_time.astimezone(local_tz)
            
            activities.append((
                row[0] or "No Title",
                row[1] or "Unknown",
                local_time.replace(tzinfo=None)
            ))
        
        logging.info(f"Found {len(activities)} activities for {device_uid}")
        return activities
        
    except Exception as e:
        logging.error(f"Error fetching activities for {device_uid}: {e}")
        return []
    finally:
        session.close()

def calculate_productivity_internal(activities):
    """Enhanced productivity calculation"""
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
        
        # Process activities with improved time distribution
        for i, (title, app, curr_time) in enumerate(activities):
            # Calculate duration (assume 2 minutes per activity if no next activity)
            if i < len(activities) - 1:
                next_time = activities[i + 1][2]
                duration = min((next_time - curr_time).total_seconds() / 60, 30)  # Max 30 minutes per activity
            else:
                duration = 2  # Default 2 minutes for last activity
            
            # Get category using FAST prediction
            category, _, _ = predict_category_fast(title, app)
            print(category)
            category_key = category.lower()
            
            # Find the hour slot for this activity
            hour_key = curr_time.replace(minute=0, second=0, microsecond=0)
            
            # Distribute duration across relevant hours
            remaining_duration = duration
            current_hour = hour_key
            
            while remaining_duration > 0 and current_hour in hourly_times:
                # Calculate how much time fits in this hour
                minutes_in_hour = min(remaining_duration, 60 - curr_time.minute if current_hour == hour_key else 60)
                
                hourly_times[current_hour][category_key] += minutes_in_hour
                remaining_duration -= minutes_in_hour
                current_hour += timedelta(hours=1)
        
        # Calculate percentages for each hour
        productivity_data = []
        for hour in hours:
            total_tracked = sum(hourly_times[hour].values())
            away_time = max(0, 60 - total_tracked)
            
            # Calculate percentages
            percentages = {
                'core productive': (hourly_times[hour].get('core productive', 0) / 60) * 100,
                'productive': (hourly_times[hour].get('productive', 0) / 60) * 100,
                'idle': (hourly_times[hour].get('idle', 0) / 60) * 100,
                'unproductive': (hourly_times[hour].get('unproductive', 0) / 60) * 100,
                'away': (away_time / 60) * 100
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
    """Fetch devices with PostgreSQL optimizations"""
    logging.info(f"Fetching devices for organization: {organization_uid}")
    session = Session()
    try:
        result = session.execute(text("""
            SELECT device_uid, user_name 
            FROM devices 
            WHERE organization_uid = :organization_uid
            ORDER BY user_name ASC
        """), {"organization_uid": organization_uid}).fetchall()
        
        devices = [{'device_uid': row[0], 'user_name': row[1] or 'Unknown User'} for row in result]
        logging.info(f"Found {len(devices)} devices for organization {organization_uid}")
        return devices
        
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
            
        # Filter out away/idle activities for more accurate working hours
        productive_activities = [
            activity for activity in activities 
            if predict_category_fast(activity[0], activity[1])[0].lower() not in ['away', 'idle']
        ]
        
        if not productive_activities:
            return "0h 0m"
        
        first_activity = productive_activities[0][2]
        last_activity = productive_activities[-1][2]
        working_duration = last_activity - first_activity
        
        # Add buffer time for the last activity
        working_duration += timedelta(minutes=10)
        
        hours, remainder = divmod(working_duration.total_seconds(), 3600)
        minutes, _ = divmod(remainder, 60)
        return f"{int(hours)}h {int(minutes)}m"
        
    except Exception as e:
        logging.error(f"Error calculating working hours: {e}")
        return "0h 0m"

@app.route('/calculate_hourly_productivity', methods=['GET'])
def calculate_hourly_productivity():
    """Enhanced hourly productivity calculation endpoint"""
    date_str = request.args.get('date')
    organization_uid = request.args.get('organization_uid')
    
    # Set up timezone
    local_tz = pytz.timezone('Asia/Kolkata')
    
    try:
        if date_str:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
        else:
            date_obj = datetime.now(local_tz).date()
    except ValueError:
        return jsonify({"error": "Invalid date format, use YYYY-MM-DD"}), 400

    if not organization_uid:
        return jsonify({"error": "organization_uid is required"}), 400

    try:
        date_for_db = date_obj.strftime('%Y-%m-%d')
        logging.info(f"Processing hourly productivity for date: {date_for_db}, organization: {organization_uid}")

        devices = get_all_device_ids(organization_uid)
        
        if not devices:
            return jsonify({"error": "No devices found for this organization"}), 404

        all_productivity_data = []

        for device in devices:
            activities = get_user_activities(device['device_uid'], date_for_db)
            
            productivity_record = calculate_productivity_internal(activities)
            working_hours = calculate_working_hours(activities)
            
            all_productivity_data.append({
                'name': device['user_name'],
                'workingHour': working_hours,
                'productivityRecord': productivity_record,
                'deviceId': device['device_uid'],
                'activityCount': len(activities)
            })
            
        logging.info(f"Successfully processed {len(all_productivity_data)} users")
        return jsonify(all_productivity_data)

    except Exception as e:
        logging.error(f"Error in hourly productivity calculation: {e}")
        return jsonify({"error": "Internal server error"}), 500

def map_category_to_productivity(category):
    """Map prediction categories to response keys"""
    mapping = {
        'Core Productive': 'Core Productivity',
        'Productive': 'Productivity',
        'Idle': 'Idle',
        'Unproductive': 'Unproductivity',
        'Away': 'Away'
    }
    return mapping.get(category, 'Idle')

def calculate_daily_productivity(activities):
    """Enhanced daily productivity calculation"""
    if not activities:
        return {
            'Core Productivity': 0.0,
            'Productivity': 0.0,
            'Unproductivity': 0.0,  
            'Idle': 0.0,
            'Away': 100.0
        }
    
    # Use time-based weighting instead of simple counting
    category_times = defaultdict(float)
    print(activities)
    total_time = 0
    
    for i, (page_title, app_name, timestamp) in enumerate(activities):
        # Calculate duration for this activity
        if i < len(activities) - 1:
            next_timestamp = activities[i + 1][2]
            duration = min((next_timestamp - timestamp).total_seconds() / 60, 30)  # Max 30 min
        else:
            duration = 2  # Default 2 minutes for last activity
        
        category, _, _ = predict_category_fast(page_title, app_name)
        mapped_category = map_category_to_productivity(category)
        
        category_times[mapped_category] += duration
        total_time += duration
    
    # Convert to percentages
    if total_time == 0:
        return {
            'Core Productivity': 0.0,
            'Productivity': 0.0,
            'Unproductivity': 0.0,
            'Idle': 0.0,
            'Away': 100.0
        }
    
    percentages = {}
    for category in ['Core Productivity', 'Productivity', 'Unproductivity', 'Idle', 'Away']:
        percentages[category] = (category_times[category] / total_time) * 100
    
    # Calculate away time (assuming 8-hour workday)
    work_minutes = 8 * 60
    active_minutes = sum(category_times.values())
    away_minutes = max(0, work_minutes - active_minutes)
    
    if away_minutes > 0:
        # Redistribute percentages to account for away time
        total_with_away = active_minutes + away_minutes
        for category in percentages:
            if category != 'Away':
                percentages[category] = (category_times[category] / total_with_away) * 100
        percentages['Away'] = (away_minutes / total_with_away) * 100
    
    return percentages

@app.route('/getUserProductivity/<device_id>', methods=['GET'])
def get_user_productivity(device_id):
    """Enhanced user productivity endpoint"""
    try:
        # Get start date from query params
        start_date_str = request.args.get('from')
        
        if not start_date_str:
            return jsonify({"error": "Missing 'from' parameter"}), 400
            
        try:
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        except ValueError:
            return jsonify({"error": "Invalid date format. Use YYYY-MM-DD."}), 400

        response_data = defaultdict(list)
        
        # Process 7-day window
        for day_offset in range(7):
            current_date = start_date + timedelta(days=day_offset)
            activities = get_user_activities(device_id, current_date.strftime('%Y-%m-%d'))
            daily_percentages = calculate_daily_productivity(activities)
            print(f'${daily_percentages} daily percentages')
            for category, value in daily_percentages.items():
                response_data[category].append(round(value, 2))

        logging.info(f"Generated 7-day productivity report for {device_id}")
        return jsonify(dict(response_data))

        
    except Exception as e:
        logging.error(f"Error in getUserProductivity: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        session = Session()
        session.execute(text("SELECT 1"))
        session.close()
        
        return jsonify({
            "status": "healthy",
            "database": "connected",
            "cache_size": len(CATEGORY_CACHE),
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logging.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/cache/clear', methods=['POST'])
def clear_cache():
    """Clear prediction cache"""
    with CACHE_LOCK:
        CATEGORY_CACHE.clear()
    return jsonify({"message": "Cache cleared successfully"})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)