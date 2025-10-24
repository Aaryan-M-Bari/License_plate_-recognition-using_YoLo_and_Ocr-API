from pyngrok import ngrok

# Your authtoken from the ngrok dashboard
AUTHTOKEN = "34VJjYIHCDnV2UvTRnKPx7dBHs8_2uNfuHVCo7e9cQ1F7eGhp"

print("Configuring pyngrok authtoken...")
ngrok.set_auth_token(AUTHTOKEN)
print("✅ pyngrok authtoken configured successfully!")
print("You can now run your api.py script.")