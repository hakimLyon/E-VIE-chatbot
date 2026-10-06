#!/bin/bash
# Quick setup script for Aura AI Django

echo "🚀 Aura AI Django - Quick Setup"
echo "================================"
echo ""

# Create virtual environment
echo "📦 Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "✅ Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Setup environment
echo "🔧 Setting up environment..."
cp .env.example .env
echo "⚠️  Please edit .env file and add your GOOGLE_API_KEY"
echo ""

# Run migrations
echo "🗄️  Running migrations..."
python manage.py migrate

# Create superuser (optional)
read -p "Create superuser for admin panel? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python manage.py createsuperuser
fi

echo ""
echo "✨ Setup complete!"
echo ""
echo "To start the development server, run:"
echo "  source venv/bin/activate"
echo "  python manage.py runserver"
echo ""
echo "Then visit http://localhost:8000"
