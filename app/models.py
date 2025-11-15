# app/models.py
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, Float, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from .database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    name = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    analyses = relationship("Analysis", back_populates="user")


class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)

    # índice por usuario (1,2,3…) para cada cuenta
    per_user_index = Column(Integer, nullable=True, index=True)

    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # nombre original del archivo
    image_filename = Column(String(255), nullable=True)

    # imagen ANOTADA en base64 (PNG) para el historial
    image_base64 = Column(Text, nullable=True)

    model_used = Column(String(50), default="best.pt")
    confidence = Column(Float, default=0.25)

    total_detections = Column(Integer, default=0)
    caries_count = Column(Integer, default=0)
    diente_retenido_count = Column(Integer, default=0)
    perdida_osea_count = Column(Integer, default=0)

    # JSON con el payload completo del análisis
    results_json = Column(Text, nullable=True)

    # JSON con dientes por patología usando sistema FDI, por ejemplo:
    # {
    #   "Caries": [16, 26],
    #   "Diente_Retenido": [38],
    #   "Perdida_Osea": []
    # }
    teeth_fdi_json = Column(Text, nullable=True)

    report_text = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="analyses")
