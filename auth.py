import os
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer
import jwt
from passlib.hash import bcrypt

SECRET_KEY = os.environ.get("SECRET_KEY", "dev_secret_change_me")
# -------------------------
# HASH PASSWORD
# -------------------------
def hash_password(password: str):
    # Encode to UTF-8 bytes and truncate to 72 bytes
    password_bytes = password[:72]
    return bcrypt.hash(password_bytes)

# -------------------------
# VERIFY PASSWORD
# -------------------------
def verify_password(password: str, hashed: str):
    # Encode to UTF-8 bytes and truncate to 72 bytes
    password_bytes = password[:72]
    return bcrypt.verify(password_bytes, hashed)

# -------------------------
# JWT CREATION / DECODING
# -------------------------
def create_jwt(user_id: int):
    token = jwt.encode({"user_id": user_id}, SECRET_KEY, algorithm="HS256")
    return token

def decode_jwt(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# -------------------------
# GET CURRENT USER
# -------------------------
def get_current_user(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload["user_id"]["user_id"]  # FIXED
    except:
        return None
