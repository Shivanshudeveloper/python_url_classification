import sqlalchemy
import psycopg2

# Define your database connection details
db_url = "postgresql://tsdbadmin:a-zi5r2T7.9GZP@c8voureiir.y3scsyzh76.tsdb.cloud.timescale.com:38528/tsdb?sslmode=require"

# Create a SQLAlchemy engine
engine = sqlalchemy.create_engine(db_url)

# Establish a connection using psycopg2 for executing raw SQL queries
connection = psycopg2.connect(db_url)

def get_user_title_by_id(userId):
    # Define an SQL query to select the user_title based on userId
    select_query = f"SELECT img_id FROM system_info WHERE id = '{userId}'"

    try:
        # Execute the query
        with connection, connection.cursor() as cursor:
            cursor.execute(select_query)

            # Fetch and print the user_title
            user_title = cursor.fetchone()
            if user_title:
                print(f"user_title for userId {userId}: {user_title[0]}")
            else:
                print(f"No user_title found for userId {userId}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Provide the userId when running the script
    user_id = input("Enter the userId: ")
    get_user_title_by_id(user_id)
    
    # Close the database connection
    connection.close()
