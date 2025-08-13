# Data Analyst Agent - Universal Web Data Analysis

A powerful, schema-driven data analysis agent that can extract and analyze data from **any URL**. Uses Google Gemini AI to intelligently understand data structures and generate custom scraping code for different types of websites and data sources.

## 🌟 Key Features

### Universal Data Support
- **HTML Tables** - Wikipedia, data tables, government sites
- **CSV Files** - Direct CSV URL processing
- **JSON APIs** - REST endpoints and JSON data
- **Excel Files** - .xlsx/.xls file processing
- **Parquet Files** - Big data format support
- **Structured Data** - JSON-LD, microdata extraction
- **Any Website** - Intelligent pattern detection and extraction

### Smart Analysis
- 🧠 **AI-Powered Code Generation** - Custom scraping code for each site
- 📊 **Automatic Schema Detection** - Understands data structure automatically  
- 🔧 **Multi-Strategy Fallbacks** - Multiple extraction methods for reliability
- 📈 **Built-in Visualizations** - Automatic plot generation
- 🧮 **Statistical Analysis** - Correlations, regressions, aggregations

## 🚀 Quick Start

1. **Set up environment:**
   ```bash
   pip install -r requirements_simple.txt
   ```

2. **Configure Google AI:**
   - Get API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Add to `.env` file:
     ```
     GOOGLE_API_KEY=your_api_key_here
     ```

3. **Run the application:**
   ```bash
   python main.py
   ```

4. **Access the web interface:**
   - Open: http://localhost:3000/frontend.html
   - Or use the API directly: http://localhost:8000

## 💡 Example Usage

### Any Wikipedia Table
```json
{
  "url": "https://en.wikipedia.org/wiki/List_of_countries_by_GDP_(nominal)",
  "tasks": [
    "What is the average GDP forecast for 2025?",
    "Which country has the highest GDP?",
    "Create a scatterplot of GDP vs Population"
  ]
}
```

### Government Data
```json
{
  "url": "https://data.gov/dataset-url",
  "tasks": [
    "What's the trend over the last 5 years?",
    "Show correlation between variables",
    "Create a visualization"
  ]
}
```

### JSON API Endpoints
```json
{
  "url": "https://api.example.com/data",
  "tasks": [
    "What's the average value?",
    "Find outliers in the data",
    "Generate summary statistics"
  ]
}
```

### CSV Files
```json
{
  "url": "https://example.com/data.csv",
  "tasks": [
    "Analyze the distribution",
    "Create a regression analysis",
    "Plot the top 10 values"
  ]
}
```

## 🏗️ Architecture

The system uses a **universal detection approach**:

1. **Auto-Detection** - Analyzes URL and content to determine data type
2. **Multi-Strategy Extraction** - Uses multiple methods for reliable data extraction
3. **AI-Powered Adaptation** - Generates custom code for each unique data source
4. **Intelligent Fallbacks** - Graceful degradation when primary methods fail

### Supported Data Types

| Type | Detection | Extraction Method |
|------|-----------|-------------------|
| HTML Tables | Table tags | pandas.read_html + BeautifulSoup |
| CSV | File extension/content | pandas.read_csv with multiple encodings |
| JSON/API | Content-Type/structure | JSON parsing + normalization |
| Excel | File extension | pandas.read_excel |
| Parquet | File extension | pandas.read_parquet |
| Structured Data | JSON-LD, microdata | Custom extraction |
| General Pages | Fallback | Pattern detection + text extraction |

## 🛠️ API Endpoints

- `POST /analyze` - Main analysis endpoint
- `GET /health` - Health check
- `GET /` - API information
- `GET /docs` - Interactive API documentation (debug mode)

## 🎯 Analysis Capabilities

### Query Types
- **Aggregations** - "What's the average/sum/count?"
- **Filtering** - "How many items meet criteria X?"
- **Correlations** - "What's the correlation between A and B?"
- **Visualizations** - "Create a scatterplot/chart"
- **Trends** - "What's the trend over time?"
- **Comparisons** - "Which item has the highest/lowest value?"

### Output Formats
- **Text Answers** - Direct responses to questions
- **Statistical Values** - Numerical results
- **Base64 Images** - Generated visualizations
- **Error Handling** - Graceful failure messages

## 🔧 Configuration

Environment variables in `.env`:

```env
# AI Configuration
GOOGLE_API_KEY=your_google_ai_key
AI_PROVIDER=google
AI_MODEL=gemini-1.5-flash

# Server Configuration  
API_HOST=127.0.0.1
API_PORT=8000
DEBUG=true

# Processing Configuration
MAX_SCHEMA_ROWS=5
REQUEST_TIMEOUT=300
MAX_FILE_SIZE=52428800
```

## 📁 Project Structure

```
├── main.py              # FastAPI application
├── config.py            # Configuration management
├── frontend.html        # Web interface
├── .env                 # Environment variables
├── requirements_simple.txt # Dependencies
└── README.md           # Documentation
```

## 🤝 Contributing

The system is designed to be easily extensible. Add new data source support by:

1. Adding detection logic in `_auto_detect_and_process()`
2. Implementing a new `_process_[format]()` method
3. Updating the LLM prompt template for the new format

## 📝 License

This project is open source and available under the MIT License.
