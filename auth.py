import os
import jwt
from fastapi import HTTPException
from passlib.context import CryptContext

SECRET_KEY = os.environ.get("SECRET_KEY", "dev_secret_change_me")

pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto"
)

# -------------------------
# HASH PASSWORD
# -------------------------
def hash_password(password: str) -> str:
    password = password[:72]  # bcrypt limit
    return pwd_context.hash(password)

# -------------------------
# VERIFY PASSWORD
# -------------------------
def verify_password(password: str, hashed_password: str) -> bool:
    return pwd_context.verify(password, hashed_password)

# -------------------------
# JWT CREATION / DECODING
# -------------------------
def create_jwt(user_id: int):
    return jwt.encode({"user_id": user_id}, SECRET_KEY, algorithm="HS256")

def decode_jwt(token: str):
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
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
        return payload["user_id"]
    except:
        return None
