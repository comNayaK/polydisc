# =============================================================
# DEPLOYMENT GUIDE — Host polymer_app as a permanent service
# =============================================================
#
# Run this ONCE. After that the app runs forever and auto-restarts
# on crashes or server reboots.
# =============================================================


# ── STEP 1: Install dependencies ──────────────────────────────

cd /path/to/polymer_app

pip install -r requirements.txt

# If you want GNN support (optional but recommended):
pip install torch-geometric
# Or follow: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

# If you want UMAP visualisation:
pip install umap-learn


# ── STEP 2: First run (trains + saves models) ─────────────────

python app.py
# Wait for all models to train and save to ./saved_models/
# This takes 10-60 min depending on GPU.
# Once done, Ctrl+C to stop. Models are now saved.


# ── STEP 3: Create systemd service ────────────────────────────

# Create the service file:
sudo nano /etc/systemd/system/polymer-app.service

# Paste this (edit paths and user):
# -------------------------------------------------------
[Unit]
Description=Polymer Discovery AI
After=network.target

[Service]
Type=simple
User=YOUR_USERNAME
WorkingDirectory=/path/to/polymer_app
ExecStart=/usr/bin/python3 /path/to/polymer_app/app.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
# -------------------------------------------------------


# ── STEP 4: Enable and start ──────────────────────────────────

sudo systemctl daemon-reload
sudo systemctl enable polymer-app
sudo systemctl start polymer-app

# Check status:
sudo systemctl status polymer-app

# View live logs:
sudo journalctl -u polymer-app -f


# ── STEP 5: (Optional) Expose on port 80 via nginx ────────────

sudo apt install nginx -y

sudo nano /etc/nginx/sites-available/polymer

# Paste this:
# -------------------------------------------------------
server {
    listen 80;
    server_name your-domain.com;   # or your IP

    location / {
        proxy_pass http://127.0.0.1:7860;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
    }
}
# -------------------------------------------------------

sudo ln -s /etc/nginx/sites-available/polymer /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx


# ── USEFUL COMMANDS ───────────────────────────────────────────

# Stop the app:
sudo systemctl stop polymer-app

# Restart after code changes:
sudo systemctl restart polymer-app

# Force retrain models:
# Edit config.py → set FORCE_RETRAIN_VAE = True or FORCE_RETRAIN_PREDICTORS = True
# Then: sudo systemctl restart polymer-app
# After retraining, set flags back to False and restart again.

# Check GPU usage:
nvidia-smi

# Check memory:
free -h
