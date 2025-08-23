from sqlalchemy import Column, Integer, String, DateTime, Text, func
from sqlalchemy.dialects.postgresql import JSON
from app.db import Base

class Meeting(Base):
    __tablename__ = "meetings"
    __table_args__ = {"extend_existing": True} 

    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    transcript = Column(Text, nullable=False)
    summary = Column(Text, nullable=False)
    tasks_json = Column(JSON, nullable=False)
