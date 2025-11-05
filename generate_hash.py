import streamlit_authenticator as stauth

# Generate password hash
hashed_password = stauth.Hasher(['admin123']).generate()[0]
print(f"Generated password hash: {hashed_password}")