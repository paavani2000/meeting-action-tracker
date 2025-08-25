from sqlalchemy import text
from app.db import Base, engine
from app.models import Meeting

print("Dropping all tables...")
Base.metadata.drop_all(engine)

# Explicitly drop leftover indexes (wrapped in text())

print("Creating all tables...")
Base.metadata.create_all(engine)

print("✅ Database initialized.")
