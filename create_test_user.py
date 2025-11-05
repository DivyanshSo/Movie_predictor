import bcrypt

# Create a new password hash
password = "test123"  # This will be the password for the test user
hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

print(f"\nTest user credentials:")
print(f"Username: testuser")
print(f"Password: {password}")
print(f"\nAdd this to your config.yaml file under credentials.usernames:")
print(f"""    testuser:
      name: Test User
      password: {hashed_password}""")