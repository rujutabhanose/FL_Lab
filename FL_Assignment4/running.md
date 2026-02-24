Terminal 1 — Start the server


cd /Users/rujutabhanose/Documents/FL_Lab/FL_Assignment4
source ../.venv/bin/activate
python server.py
Terminal 2 — Run client(s) (one or more times)


cd /Users/rujutabhanose/Documents/FL_Lab/FL_Assignment4
source ../.venv/bin/activate
python client.py
Terminal 3 — Trigger aggregation (after at least one client has run)


curl http://127.0.0.1:5001/aggregate