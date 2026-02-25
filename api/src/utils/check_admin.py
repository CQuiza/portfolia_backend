from fastapi import HTTPException, Security, status
from fastapi.security import OAuth2PasswordBearer
from src.core.config import settings
from src.utils.validations import decode_jwt

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


def check_admin_access(token: str = Security(oauth2_scheme)):
    payload = decode_jwt(token)
    if payload.get("sub") != settings.ADMIN_USERNAME:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Acceso denegado: Solo el administrador puede realizar esta acción",
        )
    return True
