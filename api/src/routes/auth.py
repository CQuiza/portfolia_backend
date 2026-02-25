from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from jose import jwt
from src.core.config import settings
from src.utils.validations import verify_password

router = APIRouter()


@router.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    # 1. Validar Usuario (Admin único)
    if form_data.username != settings.ADMIN_USERNAME:
        raise HTTPException(status_code=400, detail="Usuario incorrecto")

    # 2. Validar Password (Aquí deberías usar bcrypt para comparar el hash)
    if not verify_password(form_data.password, settings.ADMIN_PASSWORD_HASH):
        raise HTTPException(status_code=400, detail="Contraseña incorrecta")

    # 3. Crear el tiempo de expiración
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
    )

    # 4. Crear el Payload y Firmar el Token
    to_encode = {"sub": form_data.username, "exp": expire}
    encoded_jwt = jwt.encode(
        to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM
    )

    return {"access_token": encoded_jwt, "token_type": "bearer"}
