import subprocess
import sys

def run_streamlit():
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app_streamlit_proto_type.py"])
    except KeyboardInterrupt:
        print("\nStreamlit app stopped by user")
    except Exception as e:
        print(f"Error running Streamlit app: {e}")

if __name__ == "__main__":
    print("Starting Streamlit app...")
    run_streamlit()