# Modality_Specifice_Chunking

## Features

- **Multimodal Processing**: Extracts and understands text, tables, and images from PDFs
- **Visual Analysis**: Generates AI-powered descriptions of images and charts
- **Persistent Knowledge Base**: Saves processed documents for future queries
- **Docker Support**: Easy containerized deployment
- **Interactive Chat**: Command-line interface for document queries

## Installation
### Local Setup
Installation

1. Local Setup
```bash
git clone https://github.com/your-repo/multimodal-pdf-chatbot.git
cd multimodal-pdf-chatbot
```

2. Create and activate virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate    # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
echo "GOOGLE_API_KEY=your_api_key_here" > .env
```

### Docker Setup
```
docker-compose build
docker-compose up
```



