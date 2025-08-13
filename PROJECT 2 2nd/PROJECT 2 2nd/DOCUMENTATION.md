# Data Analyst Agent - Project Documentation

## 📋 Overview

The Data Analyst Agent is an advanced, AI-powered system designed to extract and analyze data from virtually any online source. It uses schema-driven extraction, intelligent code generation, and multi-format support to provide comprehensive data analysis capabilities.

## 🏗️ Architecture

### Core Components

1. **main.py** - FastAPI backend with analysis engine
2. **config.py** - Pydantic-based configuration management
3. **frontend.html** - Web interface for user interaction
4. **prompts.txt** - AI system prompts for optimal performance

### Data Processing Pipeline

```
URL Input → Auto-Detection → Schema Extraction → Code Generation → Data Extraction → Analysis → Results
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Clone or download the project
cd data-analyst-agent

# Install dependencies
pip install -r requirements_simple.txt

# Create configuration
cp .env.example .env
# Edit .env and add your API keys
```

### 2. API Key Configuration
Get a free API key from [Google AI Studio](https://makersuite.google.com/app/apikey) and add to `.env`:
```
GOOGLE_API_KEY=your_api_key_here
AI_PROVIDER=google
```

### 3. Run the Application
```bash
# Option 1: Direct execution
python main.py

# Option 2: Use startup script (Windows)
start.bat

# Option 3: Use startup script (Linux/Mac)
./start.sh
```

### 4. Access the Interface
- Web Interface: http://localhost:8000/frontend.html
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

## 🎯 Supported Data Sources

### ✅ Excellent Support
- **Wikipedia Tables** - Comprehensive extraction and analysis
- **Government Data** - CSV, Excel files from official sources
- **Sports Statistics** - Tables from sports websites
- **Financial Data** - Company listings, stock data tables
- **Research Data** - Academic datasets, Parquet files

### ⚠️ Partial Support
- **JSON APIs** - Simple REST endpoints
- **Static HTML Tables** - Non-JavaScript dependent
- **Excel Files** - Direct .xlsx/.xls downloads

### ❌ Limited Support
- **JavaScript-Heavy Sites** - Modern SPAs, dynamic content
- **Authentication Required** - Login-protected resources
- **Real-time Data** - Live dashboards, streaming data

## 🔧 Configuration Options

### Environment Variables (.env)
```bash
# AI Configuration
AI_PROVIDER=google              # "google" or "openai"
AI_MODEL=gemini-1.5-flash      # AI model to use
GOOGLE_API_KEY=your_key        # Google AI API key
OPENAI_API_KEY=your_key        # OpenAI API key (alternative)

# Server Settings
API_HOST=127.0.0.1             # Server host
API_PORT=8000                  # Server port
DEBUG=true                     # Enable debug mode

# Analysis Limits
MAX_ROWS=10000                 # Maximum rows to process
REQUEST_TIMEOUT=300            # Request timeout (seconds)
MAX_FILE_SIZE=52428800         # Max file size (50MB)

# Visualization
MAX_IMAGE_SIZE=100000          # Max plot size (100KB)
PLOT_DPI=100                   # Plot resolution
```

## 📊 Analysis Capabilities

### Statistical Operations
- Count, sum, mean, median, standard deviation
- Correlation analysis between columns
- Regression analysis with slope calculation
- Distribution analysis and percentiles

### Data Manipulation
- Filtering and conditional queries
- Grouping and aggregation
- Temporal analysis (date-based operations)
- Cross-tabulation and pivot operations

### Visualization
- Scatterplots with regression lines
- Statistical plots and charts
- Base64-encoded image output
- Optimized for web display

## 🛠️ API Usage

### POST /analyze
Analyze data from a URL with specific tasks.

**Request:**
```json
{
  "url": "https://en.wikipedia.org/wiki/List_of_countries_by_population",
  "tasks": [
    "How many countries are listed?",
    "What is the correlation between rank and population?",
    "Create a scatterplot of rank vs population"
  ]
}
```

**Response:**
```json
[
  "195 countries are listed in the table",
  "The correlation between rank and population is -0.9876",
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg..."
]
```

### GET /health
Check system status and configuration.

**Response:**
```json
{
  "status": "healthy",
  "agent": "Data Analyst Agent v2.0"
}
```

## 🧪 Testing

### Manual Testing
```bash
# Test API functionality
python test_api.py

# Test with specific URL
curl -X POST "http://localhost:8000/analyze" \
     -H "Content-Type: application/json" \
     -d '{"url": "https://example.com/data", "tasks": ["Count rows"]}'
```

### Web Interface Testing
1. Open http://localhost:8000/frontend.html
2. Enter a Wikipedia URL (e.g., List of countries by population)
3. Add analysis tasks
4. Click "Analyze Data"

## 🎭 Example Use Cases

### 1. Movie Analysis
**URL:** https://en.wikipedia.org/wiki/List_of_highest-grossing_films
**Tasks:**
- "How many movies grossed over $1 billion?"
- "What's the correlation between rank and worldwide gross?"
- "Create a scatterplot of rank vs gross with regression line"

### 2. Country Statistics
**URL:** https://en.wikipedia.org/wiki/List_of_countries_by_population
**Tasks:**
- "Which country has the largest population?"
- "What's the median population across all countries?"
- "How many countries have populations over 100 million?"

### 3. Sports Data
**URL:** https://en.wikipedia.org/wiki/List_of_FIFA_World_Cup_finals
**Tasks:**
- "How many World Cups have been held?"
- "Which country has won the most titles?"
- "What's the average attendance across all finals?"

## 🚨 Troubleshooting

### Common Issues

**Error: "Module not found"**
- Solution: Run `pip install -r requirements_simple.txt`

**Error: "API key not configured"**
- Solution: Check `.env` file and ensure `GOOGLE_API_KEY` is set

**Error: "No table found"**
- Solution: Verify URL contains tabular data, try Wikipedia URLs

**Error: "Connection timeout"**
- Solution: Check internet connection, increase `REQUEST_TIMEOUT`

### Performance Tips
- Use Wikipedia URLs for best results
- Keep tasks specific and clear
- Limit analysis to essential columns
- Monitor server logs for debugging

## 📈 Future Enhancements

### Planned Features
- Support for more file formats (JSON, XML)
- Enhanced JavaScript website support
- Database connectivity options
- Batch processing capabilities
- Custom visualization themes
- Export functionality (CSV, PDF)

### Contributing
The system is designed to be modular and extensible. Key areas for enhancement:
- Additional data source connectors
- New analysis algorithms
- Improved error handling
- Performance optimizations

## 📄 License

This project is provided as-is for educational and research purposes. Please respect website terms of service when scraping data.

---

For support or questions, refer to the API documentation at `/docs` when running the server.
