from fastapi import HTTPException, status
from jose import JWTError, jwt
from passlib.context import CryptContext
from src.core.config import settings


def decode_jwt(token: str):
    try:
        # verify sing
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )

        # if JWT valid
        return payload

    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido o expirado",
            headers={"WWW-Authenticate": "Bearer"},
        )


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password, hashed_password):
    """ """
    return pwd_context.verify(plain_password, hashed_password)
