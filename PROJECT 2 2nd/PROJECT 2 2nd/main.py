import os
import json
import base64
import io
import re
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import asyncio
import logging
import time
import traceback
import tempfile
from urllib.parse import urlparse, urljoin

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from openai import OpenAI
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

# Additional imports for enhanced content reading
try:
    import PyPDF2
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("PDF libraries not available. Install PyPDF2 and pdfplumber for PDF support.")

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("Selenium not available. Install selenium for dynamic content support.")

try:
    import aiofiles
    import httpx
    ASYNC_HTTP_AVAILABLE = True
except ImportError:
    ASYNC_HTTP_AVAILABLE = False
    print("Async HTTP libraries not available.")

# Compression and archive support
try:
    import zipfile
    import tarfile
    import gzip
    import bz2
    import lzma
    COMPRESSION_AVAILABLE = True
except ImportError:
    COMPRESSION_AVAILABLE = False
    print("Compression libraries not available.")

# Database support
try:
    import sqlite3
    import sqlalchemy
    from sqlalchemy import create_engine, text
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    print("Database libraries not available. Install sqlalchemy for database support.")

# S3 and cloud storage support
try:
    import boto3
    from botocore.exceptions import NoCredentialsError, PartialCredentialsError
    # Use compatible s3fs version
    import s3fs
    S3_AVAILABLE = True
except ImportError as e:
    S3_AVAILABLE = False
    print(f"S3 libraries not available: {e}. Install boto3 and s3fs for S3 support.")

# Additional data format support
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False
    print("Parquet support limited. Install pyarrow for full parquet support.")
try:
    import PyPDF2
    import pdfplumber
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_SUPPORT = True
except ImportError:
    SELENIUM_SUPPORT = False

try:
    import pytesseract
    from PIL import Image
    OCR_SUPPORT = True
except ImportError:
    OCR_SUPPORT = False

from config import settings

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize AI client
if settings.ai_provider.lower() == "google":
    genai.configure(api_key=settings.google_api_key)
    ai_client = genai.GenerativeModel(settings.ai_model)
    logger.info("✓ Using Google AI (Gemini)")
else:
    ai_client = OpenAI(api_key=settings.openai_api_key)
    logger.info("✓ Using OpenAI")

# FastAPI app
app = FastAPI(
    title="Data Analyst Agent - Schema-Driven Analysis",
    description="Schema-driven scraping and analysis agent for HTML tables and Parquet files",
    version="2.0.0",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class AnalysisRequest(BaseModel):
    url: str
    tasks: List[str]

class DataAnalystAgent:
    """Schema-driven data analyst agent"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': settings.user_agent
        })
        # Configure retry strategy
        self.max_retries = settings.max_retries
        self.retry_delay = settings.retry_delay
        
    async def analyze_data(self, url: str, tasks: List[str]) -> List[str]:
        """Main analysis method implementing universal data analysis"""
        logger.info(f"🔍 Starting analysis for: {url}")
        logger.info(f"📝 Tasks: {len(tasks)} items")
        
        if not url or not url.strip():
            raise HTTPException(status_code=400, detail="URL is required")
        
        if not tasks or len(tasks) == 0:
            raise HTTPException(status_code=400, detail="At least one task is required")
        
        try:
            # Step 1: Auto-detect data type and process accordingly
            df, schema = await self._auto_detect_and_process(url)
            
            logger.info(f"✓ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Validate data size
            if df.shape[0] == 0:
                raise HTTPException(status_code=400, detail="No data found at the provided URL")
            
            if df.shape[0] > settings.max_rows:
                logger.warning(f"Dataset too large ({df.shape[0]} rows), limiting to {settings.max_rows}")
                df = df.head(settings.max_rows)
            
            # Step 2: Process each task
            results = []
            for i, task in enumerate(tasks, 1):
                if not task or not task.strip():
                    results.append("Invalid task: empty or blank")
                    continue
                    
                logger.info(f"📊 Processing task {i}: {task[:50]}...")
                try:
                    result = await self._process_task(df, schema, task)
                    results.append(result)
                except Exception as task_error:
                    logger.error(f"Task {i} failed: {str(task_error)}")
                    results.append(f"Task failed: {str(task_error)}")
                
            return results
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    async def _auto_detect_and_process(self, url: str) -> tuple[pd.DataFrame, dict]:
        """Auto-detect data type and process accordingly"""
        logger.info("🔍 Auto-detecting data type")
        
        # Check URL extension and content type
        url_lower = url.lower()
        
        # Compressed files
        if any(ext in url_lower for ext in ['.zip', '.tar', '.tar.gz', '.tgz', '.tar.bz2', '.gz', '.bz2', '.xz']):
            return await self._process_compressed_file(url)
        # Database files
        elif any(ext in url_lower for ext in ['.db', '.sqlite', '.sqlite3']):
            return await self._process_database_file(url)
        # S3 URLs
        elif 's3://' in url_lower or 'amazonaws.com' in url_lower:
            return await self._process_s3_file(url)
        # PDF files
        elif any(ext in url_lower for ext in ['.pdf']):
            return await self._process_pdf(url)
        # CSV files
        elif any(ext in url_lower for ext in ['.csv', 'csv']):
            return await self._process_csv(url)
        # JSON/API
        elif any(ext in url_lower for ext in ['.json', 'json', 'api']):
            return await self._process_json_api(url)
        # Parquet files
        elif any(ext in url_lower for ext in ['.parquet', 'parquet']):
            return await self._process_parquet(url)
        # Excel files
        elif any(ext in url_lower for ext in ['.xlsx', '.xls', 'excel']):
            return await self._process_excel(url)
        # Image files
        elif any(ext in url_lower for ext in ['.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp']):
            return await self._process_image_url(url)
        else:
            # Universal HTML/webpage processing with dynamic content support
            return await self._process_universal_webpage(url)

    async def _process_pdf(self, url: str) -> tuple[pd.DataFrame, dict]:
        """Process PDF files and extract data"""
        logger.info("📄 Processing PDF file")
        
        try:
            if not PDF_AVAILABLE:
                raise Exception("PDF support not available. Install PyPDF2 and pdfplumber: pip install PyPDF2 pdfplumber")
            
            # Download PDF
            response = self.session.get(url, timeout=settings.request_timeout)
            response.raise_for_status()
            
            # Save temporarily
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                tmp_file.write(response.content)
                tmp_file.flush()
                
                # Extract text and tables from PDF
                pdf_data = await self._extract_pdf_content(tmp_file.name)
                
                # Clean up
                os.unlink(tmp_file.name)
            
            return pdf_data
            
        except Exception as e:
            raise Exception(f"Failed to process PDF: {str(e)}")
    
    async def _extract_pdf_content(self, pdf_path: str) -> tuple[pd.DataFrame, dict]:
        """Extract content from PDF using multiple methods"""
        logger.info("🔍 Extracting PDF content")
        
        try:
            # Method 1: Try pdfplumber for tables
            import pdfplumber
            
            all_tables = []
            all_text = []
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # Extract text
                    text = page.extract_text()
                    if text:
                        all_text.append(f"Page {page_num + 1}:\n{text}")
                    
                    # Extract tables
                    tables = page.extract_tables()
                    for table in tables:
                        if table and len(table) > 1:
                            all_tables.append(table)
            
            # Process extracted tables
            if all_tables:
                # Use the largest table
                best_table = max(all_tables, key=len)
                
                # Convert to DataFrame
                headers = best_table[0] if best_table else ['Column_1']
                data_rows = best_table[1:] if len(best_table) > 1 else []
                
                # Clean headers
                headers = [str(h).strip() if h else f'Column_{i+1}' for i, h in enumerate(headers)]
                
                # Create DataFrame
                df = pd.DataFrame(data_rows, columns=headers)
                
                # Clean data
                df = df.dropna(how='all').fillna('')
                
                schema = {
                    'columns': list(df.columns),
                    'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                    'sample_rows': df.head(3).to_dict('records'),
                    'shape': df.shape,
                    'data_type': 'pdf_table',
                    'source': 'PDF table extraction'
                }
                
                return df, schema
            
            # Fallback: Create data from text content
            if all_text:
                # Try to extract structured information from text
                combined_text = '\n\n'.join(all_text)
                
                # Look for numerical patterns
                numbers = re.findall(r'\d+(?:\.\d+)?', combined_text)
                dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b', combined_text)
                
                # Create structured data
                data = [{
                    'page_count': len(all_text),
                    'total_characters': len(combined_text),
                    'numbers_found': len(numbers),
                    'dates_found': len(dates),
                    'first_page_preview': all_text[0][:500] if all_text else '',
                    'content_type': 'PDF text content'
                }]
                
                df = pd.DataFrame(data)
                
                schema = {
                    'columns': list(df.columns),
                    'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                    'sample_rows': df.to_dict('records'),
                    'shape': df.shape,
                    'data_type': 'pdf_text',
                    'full_text': combined_text,
                    'source': 'PDF text extraction'
                }
                
                return df, schema
            
            raise Exception("No extractable content found in PDF")
            
        except Exception as e:
            # Final fallback using PyPDF2
            try:
                import PyPDF2
                
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text_content = []
                    
                    for page in pdf_reader.pages:
                        text_content.append(page.extract_text())
                
                combined_text = '\n'.join(text_content)
                
                data = [{
                    'pdf_pages': len(text_content),
                    'content_length': len(combined_text),
                    'content_preview': combined_text[:1000],
                    'extraction_method': 'PyPDF2'
                }]
                
                df = pd.DataFrame(data)
                
                schema = {
                    'columns': list(df.columns),
                    'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                    'sample_rows': df.to_dict('records'),
                    'shape': df.shape,
                    'data_type': 'pdf_fallback',
                    'full_text': combined_text,
                    'source': 'PyPDF2 extraction'
                }
                
                return df, schema
                
            except Exception as fallback_error:
                raise Exception(f"PDF extraction failed: {str(e)}. Fallback also failed: {str(fallback_error)}")

    async def _process_image_url(self, url: str) -> tuple[pd.DataFrame, dict]:
        """Process direct image URLs"""
        logger.info("🖼️ Processing image URL")
        
        try:
            # Download image
            response = self.session.get(url, timeout=settings.request_timeout)
            response.raise_for_status()
            
            # Basic image info
            image_size = len(response.content)
            image_type = response.headers.get('content-type', 'unknown')
            
            # Try OCR if available
            extracted_text = ""
            if OCR_SUPPORT:
                try:
                    from PIL import Image
                    import pytesseract
                    
                    # Save temporarily
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                        tmp_file.write(response.content)
                        tmp_file.flush()
                        
                        # Perform OCR
                        image = Image.open(tmp_file.name)
                        extracted_text = pytesseract.image_to_string(image)
                        
                        # Clean up
                        os.unlink(tmp_file.name)
                        
                except Exception as ocr_error:
                    logger.warning(f"OCR failed: {ocr_error}")
            
            # Create data from image analysis
            data = [{
                'image_url': url,
                'image_size_bytes': image_size,
                'image_type': image_type,
                'extracted_text': extracted_text,
                'text_length': len(extracted_text),
                'has_text': len(extracted_text.strip()) > 0,
                'analysis_method': 'OCR' if extracted_text else 'metadata_only'
            }]
            
            df = pd.DataFrame(data)
            
            schema = {
                'columns': list(df.columns),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'sample_rows': df.to_dict('records'),
                'shape': df.shape,
                'data_type': 'image_analysis',
                'extracted_text': extracted_text,
                'source': 'Direct image URL'
            }
            
            return df, schema
            
        except Exception as e:
            raise Exception(f"Failed to process image: {str(e)}")

    async def _process_universal_webpage(self, url: str) -> tuple[pd.DataFrame, dict]:
        """Universal webpage processing with dynamic content support"""
        logger.info("🌐 Processing universal webpage with comprehensive detection")
        
        try:
            # Try dynamic content first if Selenium is available
            if SELENIUM_SUPPORT:
                dynamic_result = await self._process_dynamic_content(url)
                if dynamic_result:
                    return dynamic_result
            
            # Fallback to static content processing
            return await self._process_static_content(url)
            
        except Exception as e:
            raise Exception(f"Failed to process webpage: {str(e)}")

    async def _process_dynamic_content(self, url: str) -> Optional[tuple[pd.DataFrame, dict]]:
        """Process dynamic content using Selenium"""
        logger.info("🔄 Processing dynamic content with Selenium")
        
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            
            # Setup Chrome options
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            
            # Create driver
            driver = webdriver.Chrome(options=chrome_options)
            
            try:
                # Load page
                driver.get(url)
                
                # Wait for page to load
                WebDriverWait(driver, 10).until(
                    lambda d: d.execute_script("return document.readyState") == "complete"
                )
                
                # Additional wait for dynamic content
                time.sleep(3)
                
                # Get page source after JavaScript execution
                page_source = driver.page_source
                
                # Process the rendered HTML
                soup = BeautifulSoup(page_source, 'html.parser')
                
                # Try multiple extraction strategies
                result = await self._extract_comprehensive_data(soup, url, 'dynamic')
                
                return result
                
            finally:
                driver.quit()
                
        except Exception as e:
            logger.warning(f"Dynamic content processing failed: {e}")
            return None

    async def _process_static_content(self, url: str) -> tuple[pd.DataFrame, dict]:
        """Process static webpage content"""
        logger.info("📄 Processing static webpage content")
        
        try:
            # Fetch HTML
            response = self.session.get(url, timeout=settings.request_timeout)
            response.raise_for_status()
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Comprehensive data extraction
            return await self._extract_comprehensive_data(soup, url, 'static')
            
        except Exception as e:
            raise Exception(f"Failed to process static content: {str(e)}")

    async def _extract_comprehensive_data(self, soup, url: str, method: str) -> tuple[pd.DataFrame, dict]:
        """Extract data using comprehensive detection methods"""
        logger.info(f"🔍 Comprehensive data extraction using {method} method")
        
        # Strategy 1: Look for tables
        tables = soup.find_all('table')
        if tables:
            table_result = await self._process_html_tables(soup, url)
            if table_result:
                return table_result
        
        # Strategy 2: Look for structured data (JSON-LD, microdata)
        structured_data = await self._extract_structured_data(soup)
        if structured_data:
            return structured_data
        
        # Strategy 3: Look for lists and repeated patterns
        list_data = await self._extract_list_data(soup)
        if list_data:
            return list_data
        
        # Strategy 4: Extract form data
        form_data = await self._extract_form_data(soup)
        if form_data:
            return form_data
        
        # Strategy 5: Extract images and charts with AI analysis
        image_data = await self._extract_images_and_charts(soup, url)
        if image_data:
            return image_data
        
        # Strategy 6: Extract meta information and content
        meta_data = await self._extract_meta_content(soup, url, method)
        
        return meta_data
        """Process CSV files from URLs"""
        logger.info("📄 Processing CSV file")
        
        try:
            # Try different CSV reading strategies
            strategies = [
                lambda: pd.read_csv(url),
                lambda: pd.read_csv(url, encoding='utf-8'),
                lambda: pd.read_csv(url, encoding='latin-1'),
                lambda: pd.read_csv(url, sep=';'),
                lambda: pd.read_csv(url, sep='\t'),
            ]
            
            df = None
            for strategy in strategies:
                try:
                    df = strategy()
                    if df.shape[0] > 0 and df.shape[1] > 0:
                        break
                except:
                    continue
            
            if df is None:
                raise Exception("Unable to read CSV file")
            
            # Generate schema
            schema = {
                'columns': list(df.columns),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'sample_rows': df.head(settings.max_schema_rows).to_dict('records'),
                'shape': df.shape,
                'data_type': 'csv'
            }
            
            return df, schema
            
        except Exception as e:
            raise Exception(f"Failed to process CSV file: {str(e)}")
    
    async def _process_json_api(self, url: str) -> tuple[pd.DataFrame, dict]:
        """Process JSON data or API endpoints"""
        logger.info("🌐 Processing JSON/API data")
        
        try:
            response = self.session.get(url, timeout=settings.request_timeout)
            response.raise_for_status()
            
            data = response.json()
            
            # Handle different JSON structures
            df = None
            if isinstance(data, list):
                # Array of objects
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                # Check for common API patterns
                for key in ['data', 'results', 'items', 'records']:
                    if key in data and isinstance(data[key], list):
                        df = pd.DataFrame(data[key])
                        break
                
                if df is None:
                    # Try to flatten the dict
                    df = pd.json_normalize(data)
            
            if df is None or df.empty:
                raise Exception("No tabular data found in JSON")
            
            # Generate schema
            schema = {
                'columns': list(df.columns),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'sample_rows': df.head(settings.max_schema_rows).to_dict('records'),
                'shape': df.shape,
                'data_type': 'json'
            }
            
            return df, schema
            
        except Exception as e:
            raise Exception(f"Failed to process JSON/API data: {str(e)}")
    
    async def _process_excel(self, url: str) -> tuple[pd.DataFrame, dict]:
        """Process Excel files from URLs"""
        logger.info("📊 Processing Excel file")
        
        try:
            # Download and read Excel file
            response = self.session.get(url, timeout=settings.request_timeout)
            response.raise_for_status()
            
            # Save temporarily and read
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
                tmp_file.write(response.content)
                tmp_file.flush()
                
                # Try reading different sheets
                df = pd.read_excel(tmp_file.name, sheet_name=0)
            
            if df.empty:
                raise Exception("Excel file is empty")
            
            # Generate schema
            schema = {
                'columns': list(df.columns),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'sample_rows': df.head(settings.max_schema_rows).to_dict('records'),
                'shape': df.shape,
                'data_type': 'excel'
            }
            
            return df, schema
            
        except Exception as e:
            raise Exception(f"Failed to process Excel file: {str(e)}")
    
    async def _process_html_universal(self, url: str) -> tuple[pd.DataFrame, dict]:
        """Enhanced HTML processing for any website"""
        logger.info("🌐 Processing HTML page with universal detection")
        
        try:
            # Step 1: Fetch HTML
            response = self.session.get(url, timeout=settings.request_timeout)
            response.raise_for_status()
            
            # Step 2: Try multiple detection strategies
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Strategy 1: Look for tables
            tables = soup.find_all('table')
            if tables:
                return await self._process_html_tables(soup, url)
            
            # Strategy 2: Look for structured data (JSON-LD, microdata)
            structured_data = await self._extract_structured_data(soup)
            if structured_data:
                return structured_data
            
            # Strategy 3: Look for data in divs/spans with patterns
            return await self._extract_div_data(soup, url)
            
        except Exception as e:
            raise Exception(f"Failed to process HTML: {str(e)}")
    
    async def _extract_structured_data(self, soup) -> Optional[tuple[pd.DataFrame, dict]]:
        """Extract structured data from HTML"""
        logger.info("🔍 Looking for structured data")
        
        try:
            # Look for JSON-LD
            json_scripts = soup.find_all('script', type='application/ld+json')
            for script in json_scripts:
                try:
                    data = json.loads(script.string)
                    if isinstance(data, list) and len(data) > 1:
                        df = pd.json_normalize(data)
                        if df.shape[0] > 1 and df.shape[1] > 1:
                            schema = {
                                'columns': list(df.columns),
                                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                                'sample_rows': df.head(3).to_dict('records'),
                                'shape': df.shape,
                                'data_type': 'structured'
                            }
                            return df, schema
                except:
                    continue
            
            return None
        except:
            return None
    
    async def _extract_images_and_charts(self, soup, url: str) -> Optional[tuple[pd.DataFrame, dict]]:
        """Extract and analyze images/charts using AI vision"""
        logger.info("🖼️ Extracting and analyzing images/charts with AI")
        
        try:
            # Find all images on the page
            images = soup.find_all('img')
            chart_images = []
            
            # Filter for likely chart/data images with enhanced detection
            for img in images:
                src = img.get('src', '')
                alt = img.get('alt', '').lower()
                title = img.get('title', '').lower()
                class_names = ' '.join(img.get('class', [])).lower()
                parent_text = ''
                
                # Get surrounding text context
                try:
                    parent = img.parent
                    if parent:
                        parent_text = parent.get_text()[:200].lower()
                except:
                    pass
                
                # Convert relative URLs to absolute
                if src.startswith('/'):
                    from urllib.parse import urljoin
                    src = urljoin(url, src)
                elif src.startswith('//'):
                    src = 'https:' + src
                
                # Enhanced chart indicators for GDP and economic data
                chart_indicators = [
                    'chart', 'graph', 'plot', 'data', 'gdp', 'economy', 'economic', 'statistic',
                    'bar', 'pie', 'line', 'scatter', 'histogram', 'trend', 'comparison',
                    'countries', 'ranking', 'top', 'largest', 'biggest', 'world', 'global',
                    'imf', 'forecast', 'prediction', 'estimate', 'projection', 'nominal',
                    'trillion', 'billion', 'dollar', 'usd', 'currency', 'finance', 'financial',
                    'by_country', 'by_gdp', 'economies', 'wealth', 'income', 'output',
                    'visualization', 'infographic', 'diagram', 'map', 'table'
                ]
                
                # Check multiple contexts for chart indicators
                contexts_to_check = [alt, title, class_names, parent_text, src]
                
                relevance_score = 0
                for context in contexts_to_check:
                    for indicator in chart_indicators:
                        if indicator in context:
                            # Give higher weight to more specific indicators
                            if indicator in ['gdp', 'economy', 'imf', 'forecast', 'countries']:
                                relevance_score += 3
                            elif indicator in ['chart', 'graph', 'data', 'ranking']:
                                relevance_score += 2
                            else:
                                relevance_score += 1
                
                # Also check file extensions and size indicators
                if any(ext in src.lower() for ext in ['.png', '.jpg', '.jpeg', '.svg', '.gif']):
                    relevance_score += 1
                
                # Skip very small images (likely icons)
                width = img.get('width', '0')
                height = img.get('height', '0')
                if width and height:
                    try:
                        if int(width) < 100 or int(height) < 100:
                            relevance_score -= 2
                        elif int(width) > 300 and int(height) > 200:
                            relevance_score += 2
                    except:
                        pass
                
                # Include images with any relevance score above 0
                if relevance_score > 0:
                    chart_images.append({
                        'src': src,
                        'alt': alt,
                        'title': title,
                        'relevance_score': relevance_score,
                        'context': parent_text[:100] if parent_text else ''
                    })
            
            # Sort by relevance and process top images
            chart_images.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            if not chart_images:
                logger.info("No chart images found")
                return None
            
            # Process the most relevant chart images (up to 3)
            processed_data = []
            for i, img_info in enumerate(chart_images[:3]):
                try:
                    logger.info(f"Analyzing chart image {i+1}: {img_info['src']}")
                    
                    # Download image
                    img_response = self.session.get(img_info['src'], timeout=10)
                    if img_response.status_code != 200:
                        continue
                    
                    # Convert to base64 for AI analysis
                    image_b64 = base64.b64encode(img_response.content).decode()
                    
                    # Analyze with AI (Google Gemini Vision or OpenAI Vision)
                    chart_data = await self._analyze_chart_with_ai(image_b64, img_info)
                    
                    # If AI analysis failed or returned insufficient data, try enhanced OCR
                    if not chart_data or len(chart_data.strip()) < 100:
                        logger.info(f"AI analysis insufficient, trying enhanced OCR for image {i+1}")
                        ocr_result = await self._enhanced_ocr_extraction(img_info['src'])
                        if ocr_result:
                            if chart_data:
                                chart_data += f"\n\nEnhanced OCR Results:\n{ocr_result}"
                            else:
                                chart_data = f"Enhanced OCR Extraction:\n{ocr_result}"
                    
                    if chart_data:
                        processed_data.append({
                            'image_url': img_info['src'],
                            'image_alt': img_info['alt'],
                            'image_title': img_info['title'],
                            'chart_analysis': chart_data,
                            'data_extracted': True,
                            'relevance_score': img_info['relevance_score']
                        })
                        
                except Exception as img_error:
                    logger.warning(f"Failed to process image {img_info['src']}: {img_error}")
                    continue
            
            if not processed_data:
                return None
            
            # Create enhanced DataFrame from chart analysis
            df = pd.DataFrame(processed_data)
            
            # Add extracted data as additional information for AI analysis
            chart_text_data = []
            for data_item in processed_data:
                if data_item.get('chart_analysis'):
                    # Extract any numerical data from the analysis
                    analysis_text = data_item['chart_analysis']
                    chart_text_data.append({
                        'source': data_item['image_url'],
                        'analysis': analysis_text,
                        'contains_gdp_data': 'gdp' in analysis_text.lower(),
                        'contains_forecast': any(word in analysis_text.lower() for word in ['imf', 'forecast', '2025', '2024', '2026']),
                        'text_length': len(analysis_text)
                    })
            
            # If we have chart analysis, add it as separate rows for better AI processing
            if chart_text_data:
                chart_df = pd.DataFrame(chart_text_data)
                # Combine both dataframes
                df = pd.concat([df, chart_df], ignore_index=True)
            
            schema = {
                'columns': list(df.columns),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'sample_rows': df.to_dict('records'),
                'shape': df.shape,
                'data_type': 'chart_analysis_enhanced',
                'chart_images_found': len(chart_images),
                'charts_analyzed': len(processed_data),
                'has_gdp_data': any('gdp' in str(row).lower() for row in df.to_dict('records')),
                'has_forecast_data': any(word in str(df).lower() for word in ['imf', 'forecast', '2025']),
                'extraction_note': 'Data extracted from chart images using AI vision and OCR analysis'
            }
            
            return df, schema
            
        except Exception as e:
            logger.warning(f"Image extraction failed: {e}")
            return None
    
    async def _analyze_chart_with_ai(self, image_b64: str, img_info: dict) -> Optional[str]:
        """Analyze chart image using AI vision models and OCR"""
        try:
            # First, try OCR extraction for text with enhanced processing
            ocr_text = ""
            try:
                if OCR_SUPPORT:
                    from PIL import Image, ImageEnhance, ImageFilter
                    import pytesseract
                    import io
                    
                    # Decode base64 image
                    image_data = base64.b64decode(image_b64)
                    image = Image.open(io.BytesIO(image_data))
                    
                    # Convert to RGB if needed
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    # Apply image preprocessing for better OCR
                    try:
                        # Enhance contrast
                        enhancer = ImageEnhance.Contrast(image)
                        image = enhancer.enhance(2.0)
                        
                        # Enhance sharpness
                        enhancer = ImageEnhance.Sharpness(image)
                        image = enhancer.enhance(2.0)
                        
                        # Apply slight blur to reduce noise
                        image = image.filter(ImageFilter.MedianFilter(size=3))
                        
                    except Exception as preprocess_error:
                        logger.warning(f"Image preprocessing failed: {preprocess_error}")
                    
                    # Try multiple OCR configurations for better text extraction
                    ocr_configs = [
                        '--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,%-:()[]{}$€£¥ ',
                        '--psm 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,%-:()[]{}$€£¥ ',
                        '--psm 11 --oem 3',
                        '--psm 6 --oem 3',
                        '--psm 4 --oem 3',
                        '--psm 1 --oem 3'
                    ]
                    
                    best_ocr_text = ""
                    max_text_length = 0
                    
                    for config in ocr_configs:
                        try:
                            current_text = pytesseract.image_to_string(image, config=config).strip()
                            
                            # Filter out obvious garbage and keep meaningful text
                            lines = [line.strip() for line in current_text.split('\n') if line.strip()]
                            meaningful_lines = []
                            
                            for line in lines:
                                # Keep lines that contain numbers, percentages, or meaningful words
                                if (any(char.isdigit() for char in line) or 
                                    any(word.lower() in line.lower() for word in ['gdp', 'country', 'trillion', 'billion', 'imf', 'forecast', '2024', '2025', '2026']) or
                                    len([word for word in line.split() if len(word) > 3]) >= 2):
                                    meaningful_lines.append(line)
                            
                            filtered_text = '\n'.join(meaningful_lines)
                            
                            if len(filtered_text) > max_text_length:
                                max_text_length = len(filtered_text)
                                best_ocr_text = filtered_text
                                
                        except Exception as config_error:
                            continue
                    
                    ocr_text = best_ocr_text
                    
                    # Additional post-processing to extract numerical data
                    if ocr_text:
                        # Extract numbers and potential GDP values
                        import re
                        
                        # Look for patterns like "20.5 trillion", "1,234.5", etc.
                        number_patterns = re.findall(r'\d+[.,]?\d*\s*(?:trillion|billion|million|%|USD|$)?', ocr_text, re.IGNORECASE)
                        country_patterns = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', ocr_text)
                        
                        extracted_data = []
                        if number_patterns:
                            extracted_data.append(f"Numbers found: {', '.join(number_patterns[:10])}")
                        if country_patterns:
                            extracted_data.append(f"Countries/entities found: {', '.join(country_patterns[:10])}")
                        
                        if extracted_data:
                            ocr_text += f"\n\nExtracted key data:\n" + '\n'.join(extracted_data)
                        
            except Exception as ocr_error:
                logger.warning(f"Enhanced OCR extraction failed: {ocr_error}")
            
            # Prepare enhanced analysis prompt with OCR data
            analysis_prompt = f"""
Analyze this chart/graph image and extract the numerical data it contains.

Image context: {img_info.get('alt', '')} {img_info.get('title', '')}

OCR Text extracted from image:
{ocr_text if ocr_text else "No text extracted via OCR"}

Please provide:
1. What type of chart this is (bar chart, line graph, pie chart, etc.)
2. What data it shows (e.g., GDP by country, population statistics, etc.)
3. Extract the actual numerical values and labels
4. List the data in a structured format with specific numbers
5. If this shows GDP data, identify the top 5 countries and their values
6. If there are forecasts (IMF, World Bank, etc.), extract those values specifically

Focus on extracting specific numbers, country names, values, rankings, and any forecast data.
Use both the visual chart information AND the OCR text to provide complete data extraction.
"""

            if settings.ai_provider == 'google':
                # Use Google Gemini Vision
                try:
                    import google.generativeai as genai
                    
                    # Create image data
                    image_data = {
                        'mime_type': 'image/png',
                        'data': image_b64
                    }
                    
                    # Generate analysis
                    response = genai.GenerativeModel('gemini-1.5-flash').generate_content([
                        analysis_prompt,
                        image_data
                    ])
                    
                    # Combine AI analysis with OCR text
                    ai_result = response.text
                    
                    if ocr_text and len(ocr_text.strip()) > 20:
                        combined_result = f"AI Vision Analysis:\n{ai_result}\n\nOCR Text Extraction:\n{ocr_text}"
                        return combined_result
                    else:
                        return ai_result
                    
                except Exception as gemini_error:
                    logger.warning(f"Gemini vision analysis failed: {gemini_error}")
                    
            elif settings.ai_provider == 'openai':
                # Use OpenAI Vision
                try:
                    client = OpenAI(api_key=settings.openai_api_key)
                    
                    response = client.chat.completions.create(
                        model="gpt-4-vision-preview",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": analysis_prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/png;base64,{image_b64}"}
                                    }
                                ]
                            }
                        ],
                        max_tokens=1000
                    )
                    
                    # Combine AI analysis with OCR text
                    ai_result = response.choices[0].message.content
                    
                    if ocr_text and len(ocr_text.strip()) > 20:
                        combined_result = f"AI Vision Analysis:\n{ai_result}\n\nOCR Text Extraction:\n{ocr_text}"
                        return combined_result
                    else:
                        return ai_result
                    
                except Exception as openai_error:
                    logger.warning(f"OpenAI vision analysis failed: {openai_error}")
            
            # Fallback: return OCR text if AI fails
            if ocr_text and len(ocr_text.strip()) > 10:
                return f"OCR Text Extraction (AI analysis unavailable):\n{ocr_text}"
            
            return None
            
        except Exception as e:
            logger.warning(f"AI chart analysis failed: {e}")
            return None
    
    async def _enhanced_ocr_extraction(self, image_url: str) -> Optional[str]:
        """Standalone enhanced OCR extraction for images"""
        try:
            logger.info(f"🔍 Enhanced OCR processing for: {image_url}")
            
            if not OCR_SUPPORT:
                return None
            
            # Download image
            response = self.session.get(image_url, timeout=10)
            if response.status_code != 200:
                return None
            
            from PIL import Image, ImageEnhance, ImageFilter, ImageOps
            import pytesseract
            import io
            
            # Load image
            image = Image.open(io.BytesIO(response.content))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Try multiple preprocessing techniques
            preprocessing_methods = [
                lambda img: img,  # Original
                lambda img: ImageEnhance.Contrast(img).enhance(2.5),  # High contrast
                lambda img: ImageOps.grayscale(img),  # Grayscale
                lambda img: img.filter(ImageFilter.SHARPEN),  # Sharpen
                lambda img: ImageEnhance.Brightness(img).enhance(1.5),  # Brighten
            ]
            
            best_text = ""
            max_meaningful_content = 0
            
            for preprocess_func in preprocessing_methods:
                try:
                    processed_img = preprocess_func(image)
                    
                    # Try different OCR configs
                    ocr_configs = [
                        '--psm 6 --oem 3',
                        '--psm 3 --oem 3', 
                        '--psm 11 --oem 3',
                        '--psm 4 --oem 3',
                        '--psm 1 --oem 3'
                    ]
                    
                    for config in ocr_configs:
                        try:
                            text = pytesseract.image_to_string(processed_img, config=config)
                            
                            # Score the text quality
                            meaningful_score = 0
                            lines = [line.strip() for line in text.split('\n') if line.strip()]
                            
                            for line in lines:
                                # Score based on content relevance
                                if any(word in line.lower() for word in ['gdp', 'country', 'trillion', 'billion', 'imf', 'forecast']):
                                    meaningful_score += 10
                                elif any(char.isdigit() for char in line):
                                    meaningful_score += 5
                                elif len(line.split()) >= 3:
                                    meaningful_score += 2
                            
                            if meaningful_score > max_meaningful_content:
                                max_meaningful_content = meaningful_score
                                best_text = text
                                
                        except Exception:
                            continue
                            
                except Exception:
                    continue
            
            if best_text.strip():
                # Post-process the best text
                lines = [line.strip() for line in best_text.split('\n') if line.strip()]
                
                # Filter and enhance lines
                enhanced_lines = []
                for line in lines:
                    # Keep lines with numbers, economic terms, or country names
                    if (any(char.isdigit() for char in line) or
                        any(word.lower() in line.lower() for word in [
                            'gdp', 'country', 'trillion', 'billion', 'million', 'imf', 'forecast',
                            'united states', 'china', 'japan', 'germany', 'india', 'uk', 'france',
                            'italy', 'brazil', 'canada', 'russia', 'economy', 'nominal', '2024', '2025'
                        ]) or
                        len([word for word in line.split() if len(word) > 3]) >= 2):
                        enhanced_lines.append(line)
                
                if enhanced_lines:
                    return '\n'.join(enhanced_lines)
            
            return None
            
        except Exception as e:
            logger.warning(f"Enhanced OCR extraction failed for {image_url}: {e}")
            return None
    
    async def _extract_div_data(self, soup, url: str) -> tuple[pd.DataFrame, dict]:
        """Extract data from div/span elements with patterns"""
        logger.info("🔍 Extracting data from div/span patterns")
        
        try:
            # Look for repeated patterns
            data_rows = []
            
            # Common patterns for data
            patterns = [
                {'class': re.compile(r'.*item.*', re.I)},
                {'class': re.compile(r'.*row.*', re.I)},
                {'class': re.compile(r'.*entry.*', re.I)},
                {'class': re.compile(r'.*record.*', re.I)},
            ]
            
            for pattern in patterns:
                elements = soup.find_all('div', pattern)
                if len(elements) > 5:  # Need at least 5 similar elements
                    # Try to extract structured data
                    for elem in elements[:20]:  # Limit to first 20
                        row_data = {}
                        # Extract text from child elements
                        for child in elem.find_all(['span', 'div', 'p'])[:10]:
                            if child.get_text(strip=True):
                                key = child.get('class', ['data'])[0] if child.get('class') else 'data'
                                row_data[key] = child.get_text(strip=True)
                        
                        if len(row_data) > 1:
                            data_rows.append(row_data)
                    
                    if len(data_rows) > 3:
                        break
            
            if not data_rows:
                # Final fallback - create minimal data
                return await self._create_minimal_data(soup, url)
            
            df = pd.DataFrame(data_rows)
            schema = {
                'columns': list(df.columns),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'sample_rows': df.head(3).to_dict('records'),
                'shape': df.shape,
                'data_type': 'extracted'
            }
            
            return df, schema
            
        except Exception as e:
            return await self._create_minimal_data(soup, url)
    
    async def _create_minimal_data(self, soup, url: str) -> tuple[pd.DataFrame, dict]:
        """Create minimal data from webpage content"""
        logger.info("📝 Creating minimal data extraction")
        
        try:
            # Extract basic page information
            title = soup.find('title')
            headings = soup.find_all(['h1', 'h2', 'h3'])
            paragraphs = soup.find_all('p')
            
            data = [{
                'url': url,
                'title': title.get_text(strip=True) if title else 'No title',
                'headings_count': len(headings),
                'paragraphs_count': len(paragraphs),
                'content_length': len(soup.get_text()),
                'first_heading': headings[0].get_text(strip=True) if headings else '',
                'first_paragraph': paragraphs[0].get_text(strip=True)[:200] if paragraphs else ''
            }]
            
            df = pd.DataFrame(data)
            schema = {
                'columns': list(df.columns),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'sample_rows': df.to_dict('records'),
                'shape': df.shape,
                'data_type': 'minimal'
            }
            
            return df, schema
            
        except Exception as e:
            raise Exception(f"Failed to create minimal data: {str(e)}")
    
    async def _extract_list_data(self, soup) -> Optional[tuple[pd.DataFrame, dict]]:
        """Extract data from lists and repeated patterns"""
        logger.info("📝 Extracting list data")
        
        try:
            # Look for ordered and unordered lists
            lists = soup.find_all(['ul', 'ol'])
            
            for list_elem in lists:
                items = list_elem.find_all('li')
                if len(items) > 5:  # Need meaningful data
                    list_data = []
                    for item in items:
                        text = item.get_text(strip=True)
                        if text and len(text) > 10:  # Meaningful content
                            list_data.append({'item': text})
                    
                    if len(list_data) > 3:
                        df = pd.DataFrame(list_data)
                        schema = {
                            'columns': list(df.columns),
                            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                            'sample_rows': df.head(3).to_dict('records'),
                            'shape': df.shape,
                            'data_type': 'list'
                        }
                        return df, schema
            
            return None
            
        except Exception as e:
            logger.warning(f"List extraction failed: {e}")
            return None
    
    async def _extract_form_data(self, soup) -> Optional[tuple[pd.DataFrame, dict]]:
        """Extract data from forms"""
        logger.info("📝 Extracting form data")
        
        try:
            forms = soup.find_all('form')
            
            for form in forms:
                inputs = form.find_all(['input', 'select', 'textarea'])
                if len(inputs) > 3:
                    form_data = []
                    for input_elem in inputs:
                        name = input_elem.get('name', '')
                        value = input_elem.get('value', '')
                        input_type = input_elem.get('type', input_elem.name)
                        
                        if name:
                            form_data.append({
                                'field_name': name,
                                'field_type': input_type,
                                'default_value': value
                            })
                    
                    if len(form_data) > 2:
                        df = pd.DataFrame(form_data)
                        schema = {
                            'columns': list(df.columns),
                            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                            'sample_rows': df.to_dict('records'),
                            'shape': df.shape,
                            'data_type': 'form'
                        }
                        return df, schema
            
            return None
            
        except Exception as e:
            logger.warning(f"Form extraction failed: {e}")
            return None
    
    async def _extract_meta_content(self, soup, url: str, method: str) -> tuple[pd.DataFrame, dict]:
        """Extract meta information and content as fallback"""
        logger.info("📝 Extracting meta content as fallback")
        
        try:
            # Extract page metadata
            title = soup.find('title')
            meta_tags = soup.find_all('meta')
            headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            
            # Collect meta information
            meta_data = {
                'url': url,
                'extraction_method': method,
                'page_title': title.get_text(strip=True) if title else 'No title',
                'meta_tags_count': len(meta_tags),
                'headings_count': len(headings),
                'content_length': len(soup.get_text()),
                'has_tables': len(soup.find_all('table')) > 0,
                'has_forms': len(soup.find_all('form')) > 0,
                'has_lists': len(soup.find_all(['ul', 'ol'])) > 0,
                'images_count': len(soup.find_all('img')),
                'links_count': len(soup.find_all('a'))
            }
            
            # Add first few headings
            for i, heading in enumerate(headings[:5]):
                meta_data[f'heading_{i+1}'] = heading.get_text(strip=True)
            
            # Add meta description if available
            description = soup.find('meta', attrs={'name': 'description'})
            if description:
                meta_data['meta_description'] = description.get('content', '')
            
            df = pd.DataFrame([meta_data])
            
            schema = {
                'columns': list(df.columns),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'sample_rows': df.to_dict('records'),
                'shape': df.shape,
                'data_type': 'meta_content'
            }
            
            return df, schema
            
        except Exception as e:
            raise Exception(f"Failed to extract meta content: {str(e)}")
    
    async def _process_html_tables(self, soup, url: str) -> tuple[pd.DataFrame, dict]:
        """Process HTML tables (enhanced version of original method)"""
        logger.info("📊 Processing HTML tables")
        
        try:
            # Find the best table
            tables = soup.find_all('table')
            best_table = None
            max_data_score = 0
            
            for table in tables:
                # Score based on size and content
                rows = table.find_all('tr')
                if len(rows) < 2:
                    continue
                
                # Check data richness - fix the indexing error
                first_row = rows[0]
                cells_in_first_row = first_row.find_all(['td', 'th'])
                if len(cells_in_first_row) == 0:
                    continue
                    
                data_score = len(rows) * len(cells_in_first_row)
                if data_score > max_data_score:
                    max_data_score = data_score
                    best_table = table
            
            if not best_table:
                raise Exception("No suitable table found")
            
            # Extract schema from best table
            schema = self._extract_table_schema(best_table)
            
            # Generate and execute scraping code
            scraping_code = await self._generate_scraping_code(schema, url)
            df = self._execute_scraping_code(scraping_code, url)
            
            return df, schema
            
        except Exception as e:
            raise Exception(f"Failed to process HTML tables: {str(e)}")
    
    async def _process_parquet(self, url: str) -> tuple[pd.DataFrame, dict]:
        """Process Parquet file and extract schema"""
        logger.info("📄 Processing Parquet file")
        
        try:
            # Read parquet file
            df = pd.read_parquet(url)
            
            # Extract schema
            schema = {
                'columns': list(df.columns),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'sample_rows': df.head(settings.max_schema_rows).to_dict('records'),
                'shape': df.shape
            }
            
            return df, schema
            
        except Exception as e:
            raise Exception(f"Failed to process Parquet file: {str(e)}")
    
    async def _process_compressed_file(self, url: str) -> tuple[pd.DataFrame, dict]:
        """Process compressed files (zip, tar, gz, etc.)"""
        logger.info("📦 Processing compressed file")
        
        try:
            if not COMPRESSION_AVAILABLE:
                raise Exception("Compression support not available")
            
            # Download the compressed file
            response = self.session.get(url, timeout=settings.request_timeout)
            response.raise_for_status()
            
            # Save temporarily
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(response.content)
                tmp_file.flush()
                
                # Extract and process
                extracted_data = await self._extract_compressed_content(tmp_file.name, url)
                
                # Clean up
                os.unlink(tmp_file.name)
                
                return extracted_data
                
        except Exception as e:
            raise Exception(f"Failed to process compressed file: {str(e)}")
    
    async def _extract_compressed_content(self, file_path: str, original_url: str) -> tuple[pd.DataFrame, dict]:
        """Extract content from compressed files"""
        logger.info("🔍 Extracting compressed content")
        
        url_lower = original_url.lower()
        all_dataframes = []
        
        try:
            # Handle different compression types
            if '.zip' in url_lower:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    for file_name in zip_ref.namelist():
                        if file_name.endswith(('.csv', '.json', '.xlsx', '.parquet')):
                            with zip_ref.open(file_name) as extracted_file:
                                df = await self._process_extracted_file(extracted_file, file_name)
                                if df is not None:
                                    all_dataframes.append((df, file_name))
            
            elif any(ext in url_lower for ext in ['.tar', '.tgz', '.tar.gz', '.tar.bz2']):
                mode = 'r:gz' if '.gz' in url_lower or '.tgz' in url_lower else 'r:bz2' if '.bz2' in url_lower else 'r'
                with tarfile.open(file_path, mode) as tar_ref:
                    for member in tar_ref.getmembers():
                        if member.isfile() and any(member.name.endswith(ext) for ext in ['.csv', '.json', '.xlsx', '.parquet']):
                            extracted_file = tar_ref.extractfile(member)
                            if extracted_file:
                                df = await self._process_extracted_file(extracted_file, member.name)
                                if df is not None:
                                    all_dataframes.append((df, member.name))
            
            elif '.gz' in url_lower:
                with gzip.open(file_path, 'rb') as gz_file:
                    # Assume it's a CSV for now
                    df = pd.read_csv(gz_file)
                    all_dataframes.append((df, 'compressed_file.csv'))
            
            # Combine all dataframes or use the largest one
            if all_dataframes:
                if len(all_dataframes) == 1:
                    df, source_file = all_dataframes[0]
                else:
                    # Use the largest dataframe
                    df, source_file = max(all_dataframes, key=lambda x: x[0].shape[0])
                
                schema = {
                    'columns': list(df.columns),
                    'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                    'sample_rows': df.head(3).to_dict('records'),
                    'shape': df.shape,
                    'data_type': 'compressed_archive',
                    'source_file': source_file,
                    'total_files_found': len(all_dataframes),
                    'source': f'Compressed archive: {original_url}'
                }
                
                return df, schema
            
            raise Exception("No processable files found in compressed archive")
            
        except Exception as e:
            raise Exception(f"Failed to extract compressed content: {str(e)}")
    
    async def _process_extracted_file(self, file_obj, file_name: str) -> pd.DataFrame:
        """Process individual files from compressed archives"""
        try:
            if file_name.endswith('.csv'):
                return pd.read_csv(file_obj)
            elif file_name.endswith('.json'):
                return pd.read_json(file_obj)
            elif file_name.endswith(('.xlsx', '.xls')):
                return pd.read_excel(file_obj)
            elif file_name.endswith('.parquet'):
                return pd.read_parquet(file_obj)
            return None
        except Exception:
            return None
    
    async def _process_database_file(self, url: str) -> tuple[pd.DataFrame, dict]:
        """Process database files (SQLite, etc.)"""
        logger.info("🗄️ Processing database file")
        
        try:
            if not DATABASE_AVAILABLE:
                raise Exception("Database support not available. Install sqlalchemy: pip install sqlalchemy")
            
            # Download the database file
            response = self.session.get(url, timeout=settings.request_timeout)
            response.raise_for_status()
            
            # Save temporarily
            with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
                tmp_file.write(response.content)
                tmp_file.flush()
                
                # Process database
                db_data = await self._extract_database_content(tmp_file.name)
                
                # Clean up
                os.unlink(tmp_file.name)
                
                return db_data
                
        except Exception as e:
            raise Exception(f"Failed to process database file: {str(e)}")
    
    async def _extract_database_content(self, db_path: str) -> tuple[pd.DataFrame, dict]:
        """Extract content from database files"""
        logger.info("🔍 Extracting database content")
        
        try:
            # Connect to SQLite database
            conn = sqlite3.connect(db_path)
            
            # Get all table names
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            if not tables:
                raise Exception("No tables found in database")
            
            # Get the largest table or first table
            largest_table = None
            largest_size = 0
            
            for table_name, in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                size = cursor.fetchone()[0]
                if size > largest_size:
                    largest_size = size
                    largest_table = table_name
            
            if not largest_table:
                raise Exception("No data found in database tables")
            
            # Read the largest table
            df = pd.read_sql_query(f"SELECT * FROM {largest_table}", conn)
            
            # Get table info
            cursor.execute(f"PRAGMA table_info({largest_table})")
            columns_info = cursor.fetchall()
            
            conn.close()
            
            schema = {
                'columns': list(df.columns),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'sample_rows': df.head(3).to_dict('records'),
                'shape': df.shape,
                'data_type': 'database',
                'table_name': largest_table,
                'total_tables': len(tables),
                'table_info': columns_info,
                'source': 'SQLite database'
            }
            
            return df, schema
            
        except Exception as e:
            raise Exception(f"Failed to extract database content: {str(e)}")
    
    async def _process_s3_file(self, url: str) -> tuple[pd.DataFrame, dict]:
        """Process S3 files (especially parquet)"""
        logger.info("☁️ Processing S3 file")
        
        try:
            if not S3_AVAILABLE:
                raise Exception("S3 support not available. Install boto3 and s3fs: pip install boto3 s3fs")
            
            # Parse S3 URL
            if url.startswith('s3://'):
                # Direct S3 URL
                s3_path = url
            else:
                # Extract S3 path from AWS URL
                import re
                match = re.search(r'https://([^.]+)\.s3[^/]*\.amazonaws\.com/(.+)', url)
                if match:
                    bucket, key = match.groups()
                    s3_path = f's3://{bucket}/{key}'
                else:
                    raise Exception("Invalid S3 URL format")
            
            # Try to read as parquet first
            try:
                if s3_path.lower().endswith('.parquet'):
                    df = pd.read_parquet(s3_path)
                elif s3_path.lower().endswith('.csv'):
                    df = pd.read_csv(s3_path)
                elif s3_path.lower().endswith('.json'):
                    df = pd.read_json(s3_path)
                else:
                    # Try parquet by default
                    df = pd.read_parquet(s3_path)
                
                schema = {
                    'columns': list(df.columns),
                    'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                    'sample_rows': df.head(3).to_dict('records'),
                    'shape': df.shape,
                    'data_type': 's3_file',
                    's3_path': s3_path,
                    'source': 'S3 cloud storage'
                }
                
                return df, schema
                
            except Exception as read_error:
                raise Exception(f"Failed to read S3 file: {str(read_error)}")
                
        except Exception as e:
            raise Exception(f"Failed to process S3 file: {str(e)}")
    
    async def _process_html(self, url: str) -> tuple[pd.DataFrame, dict]:
        """Process HTML page using schema-driven approach"""
        logger.info("🌐 Processing HTML page")
        
        try:
            # Step 1: Fetch HTML
            response = self.session.get(url, timeout=settings.request_timeout)
            response.raise_for_status()
            
            # Step 2: Extract table schema
            soup = BeautifulSoup(response.content, 'html.parser')
            table = soup.find('table')
            
            if not table:
                raise Exception("No table found in HTML")
            
            # Extract schema from first table
            schema = self._extract_table_schema(table)
            
            # Step 3: Generate scraping code using LLM
            scraping_code = await self._generate_scraping_code(schema, url)
            
            # Step 4: Execute scraping code
            df = self._execute_scraping_code(scraping_code, url)
            
            return df, schema
            
        except Exception as e:
            raise Exception(f"Failed to process HTML: {str(e)}")
    
    def _extract_table_schema(self, table) -> dict:
        """Extract lightweight schema from HTML table"""
        logger.info("🔍 Extracting table schema")
        
        # Get headers - try different approaches
        headers = []
        
        # Try to find header row (th elements first)
        header_candidates = table.find_all('tr')
        for row in header_candidates[:3]:  # Check first 3 rows
            potential_headers = []
            
            # First try th elements
            th_elements = row.find_all('th')
            if th_elements:
                potential_headers = [th.get_text(strip=True) for th in th_elements]
            else:
                # Fallback to td elements in first row
                td_elements = row.find_all('td')
                if td_elements:
                    potential_headers = [td.get_text(strip=True) for td in td_elements]
            
            # Use the first non-empty set of headers found
            if potential_headers and any(h.strip() for h in potential_headers):
                headers = potential_headers
                break
        
        # If no good headers found, create generic ones
        if not headers:
            # Count columns from first data row
            first_data_row = table.find('tr')
            if first_data_row:
                cell_count = len(first_data_row.find_all(['td', 'th']))
                headers = [f'Column_{i+1}' for i in range(cell_count)]
        
        if not headers:
            headers = ['Column_1']  # Absolute fallback
        
        # Get sample rows - skip potential header rows
        sample_rows = []
        all_rows = table.find_all('tr')
        
        # Start from row 1 (skip potential header)
        data_rows = all_rows[1:] if len(all_rows) > 1 else all_rows
        
        for row in data_rows[:settings.max_schema_rows]:
            cells = row.find_all(['td', 'th'])
            if len(cells) >= len(headers):
                row_data = [cell.get_text(strip=True) for cell in cells[:len(headers)]]
                # Only add rows with some actual content
                if any(cell.strip() for cell in row_data):
                    sample_rows.append(dict(zip(headers, row_data)))
        
        # If no sample rows found, create a dummy one
        if not sample_rows:
            sample_rows = [dict(zip(headers, [''] * len(headers)))]
        
        schema = {
            'columns': headers,
            'sample_rows': sample_rows,
            'table_count': len(table.find_parent().find_all('table')) if table.find_parent() else 1,
            'data_type': 'html_table'
        }
        
        logger.info(f"✓ Schema extracted: {len(headers)} columns, {len(sample_rows)} sample rows")
        return schema
    
    async def _generate_scraping_code(self, schema: dict, url: str) -> str:
        """Generate scraping code using LLM based on schema"""
        logger.info("🤖 Generating scraping code with LLM")
        
        prompt_template = """
You are a Python data scraping expert. Generate ONLY the Python code to scrape data from ANY type of webpage.

URL: {url}
Data Type: {data_type}
Table Schema (if applicable):
- Columns: {columns}
- Sample rows: {sample_rows}

Requirements:
1. AUTO-DETECT the data type and use appropriate extraction method:
   - For tables: Use pandas.read_html() or BeautifulSoup table parsing
   - For structured data: Look for JSON-LD, microdata, or API endpoints
   - For lists/divs: Extract repeated patterns
   - For any other content: Extract meaningful text data
2. The final DataFrame must be assigned to variable 'df'
3. Clean and process data appropriately (handle different data types)
4. Ensure proper column names and data structure
5. Handle various website structures and layouts
6. Return ONLY executable Python code, no explanations

Universal scraping template:
```python
import pandas as pd
import requests
from bs4 import BeautifulSoup
import numpy as np
import json
import re

try:
    response = requests.get('{url_example}')
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Strategy 1: Try pandas.read_html for tables
    try:
        tables = pd.read_html('{url_example}')
        for table in tables:
            if len(table.columns) >= 2 and len(table) > 2:
                df = table
                break
        else:
            df = tables[0] if tables else None
    except:
        df = None
    
    # Strategy 2: Manual table parsing if pandas failed
    if df is None:
        tables = soup.find_all('table')
        for table in tables:
            try:
                headers = [th.get_text(strip=True) for th in table.find('tr').find_all(['th', 'td'])]
                if len(headers) < 2:
                    continue
                    
                rows = []
                for tr in table.find_all('tr')[1:]:
                    cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
                    if len(cells) >= len(headers):
                        rows.append(cells[:len(headers)])
                
                if len(rows) > 2:
                    df = pd.DataFrame(rows, columns=headers)
                    break
            except:
                continue
    
    # Strategy 3: Look for structured data (JSON-LD)
    if df is None:
        json_scripts = soup.find_all('script', type='application/ld+json')
        for script in json_scripts:
            try:
                data = json.loads(script.string)
                if isinstance(data, list) and len(data) > 1:
                    df = pd.json_normalize(data)
                    break
            except:
                continue
    
    # Strategy 4: Extract from repeated div/span patterns
    if df is None:
        patterns = [
            soup.find_all('div', class_=re.compile(r'.*item.*', re.I)),
            soup.find_all('div', class_=re.compile(r'.*row.*', re.I)),
            soup.find_all('li')
        ]
        
        for pattern_elements in patterns:
            if len(pattern_elements) > 3:
                data_rows = []
                for elem in pattern_elements[:20]:
                    row_data = {{}}
                    text_content = elem.get_text(strip=True)
                    if text_content:
                        # Try to extract structured info
                        children = elem.find_all(['span', 'div', 'p', 'a'])[:5]
                        for i, child in enumerate(children):
                            if child.get_text(strip=True):
                                key = child.get('class', [f'field_{i}'])[0] if child.get('class') else f'field_{i}'
                                row_data[key] = child.get_text(strip=True)
                        
                        if not row_data:
                            row_data = {{'content': text_content[:200]}}
                        
                        data_rows.append(row_data)
                
                if len(data_rows) > 2:
                    df = pd.DataFrame(data_rows)
                    break
    
    # Strategy 5: Fallback - create basic page info
    if df is None:
        title = soup.find('title')
        headings = soup.find_all(['h1', 'h2', 'h3'])
        
        df = pd.DataFrame([{{
            'url': '{url_example}',
            'title': title.get_text(strip=True) if title else 'No title',
            'headings_count': len(headings),
            'content_preview': soup.get_text()[:500].strip(),
            'page_type': 'general_webpage'
        }}])

except Exception as e:
    # Final fallback
    df = pd.DataFrame([{{'error': str(e), 'url': '{url_example}'}}])

# Clean the dataframe
if df is not None and not df.empty:
    # Clean column names
    df.columns = [str(col).strip() for col in df.columns]
    
    # Try to convert numeric columns
    for col in df.columns:
        if df[col].dtype == 'object':
            numeric_col = pd.to_numeric(df[col].astype(str).str.replace(r'[^\\d.-]', '', regex=True), errors='ignore')
            if not numeric_col.equals(df[col]) and not numeric_col.isna().all():
                df[col] = numeric_col
```

Generate the scraping code for this URL:
"""
        
        prompt = prompt_template.replace('{url}', url).replace('{data_type}', schema.get('data_type', 'unknown')).replace('{columns}', str(schema.get('columns', []))).replace('{sample_rows}', str(schema.get('sample_rows', [])[:2])).replace('{url_example}', url)
        
        try:
            if settings.ai_provider.lower() == "google":
                response = ai_client.generate_content(prompt)
                code = response.text
            else:
                response = ai_client.chat.completions.create(
                    model=settings.ai_model,
                    messages=[
                        {"role": "system", "content": "You are a Python data scraping expert. Generate only executable Python code."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=settings.ai_max_tokens
                )
                code = response.choices[0].message.content
            
            # Clean the code
            code = self._clean_generated_code(code)
            logger.info("✓ Scraping code generated")
            return code
            
        except Exception as e:
            logger.error(f"Failed to generate scraping code: {e}")
            # Enhanced fallback code for Wikipedia tables
            fallback_code = '''
import pandas as pd
import requests
from bs4 import BeautifulSoup
import numpy as np

try:
    tables = pd.read_html('{url}')
    df = None
    
    # Find table with most data
    for table in tables:
        if len(table.columns) >= 3 and len(table) > 10:
            df = table
            break
    
    if df is None and tables:
        df = tables[0]
        
    # Clean column names
    if df is not None:
        df.columns = df.columns.astype(str)
        
        # Clean numeric columns
        for col in df.columns:
            if df[col].dtype == 'object':
                numeric_col = pd.to_numeric(df[col].astype(str).str.replace(r'[^\\d.-]', '', regex=True), errors='ignore')
                if not numeric_col.equals(df[col]):
                    df[col] = numeric_col
                    
except Exception as e:
    # Manual parsing
    response = requests.get('{url}')
    soup = BeautifulSoup(response.content, 'html.parser')
    
    tables = soup.find_all('table')
    for table in tables:
        try:
            headers = [th.get_text(strip=True) for th in table.find('tr').find_all(['th', 'td'])]
            if len(headers) < 2:
                continue
                
            rows = []
            for tr in table.find_all('tr')[1:]:
                cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
                if len(cells) >= len(headers):
                    rows.append(cells[:len(headers)])
            
            if len(rows) > 10:
                df = pd.DataFrame(rows, columns=headers)
                break
        except:
            continue

if df is None:
    df = pd.DataFrame([['No data']], columns=['Error'])
'''
            return fallback_code.format(url=url)
    
    def _clean_generated_code(self, code: str) -> str:
        """Clean and validate generated code"""
        # Remove markdown code blocks
        code = re.sub(r'```python\n?', '', code)
        code = re.sub(r'```\n?', '', code)
        
        # Ensure required imports
        imports = [
            "import pandas as pd",
            "import requests", 
            "from bs4 import BeautifulSoup",
            "import numpy as np"
        ]
        
        for imp in imports:
            if imp not in code:
                code = f"{imp}\n{code}"
        
        return code.strip()
    
    def _execute_scraping_code(self, code: str, url: str) -> pd.DataFrame:
        """Safely execute the generated scraping code"""
        logger.info("⚙️ Executing scraping code")
        
        # Create execution environment
        exec_globals = {
            'pd': pd,
            'requests': requests,
            'BeautifulSoup': BeautifulSoup,
            'np': np,
            'url': url,
            'json': json,
            're': re
        }
        
        try:
            # Execute the code
            exec(code, exec_globals)
            
            # Get the dataframe from execution
            df = exec_globals.get('df')
            
            if df is None:
                raise ValueError("Code execution didn't produce a dataframe")
            
            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"Expected DataFrame, got {type(df)}")
            
            if df.empty:
                raise ValueError("DataFrame is empty")
                
            # Clean and validate the dataframe
            df = self._clean_execution_result(df)
            
            logger.info(f"✓ Code execution successful: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Code execution failed: {e}")
            raise Exception(f"Failed to execute scraping code: {str(e)}")
    
    def _clean_execution_result(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate dataframe from code execution"""
        logger.info(f"🧹 Cleaning execution result: {df.shape}")
        
        if df is None or df.empty:
            raise ValueError("DataFrame is empty or None")
        
        # Remove completely empty rows and columns
        df = df.dropna(how='all').loc[:, df.notna().any()]
        
        if df.empty:
            raise ValueError("DataFrame is empty after removing null rows/columns")
        
        # Clean column names
        df.columns = [str(col).strip() for col in df.columns]
        
        # Handle duplicate column names
        if df.columns.duplicated().any():
            logger.warning("Found duplicate column names, making them unique")
            df.columns = pd.io.common.dedup_names(df.columns, is_potential_multiindex=False)
        
        # Clean and optimize data types
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    # Clean whitespace
                    df[col] = df[col].astype(str).str.strip()
                    df[col] = df[col].replace(r'\s+', ' ', regex=True)  # Multiple spaces to single
                    df[col] = df[col].replace('', pd.NA)  # Empty strings to NaN
                    
                    # Try to convert to numeric if it looks like numbers
                    cleaned_series = df[col].astype(str).str.replace(r'[^\d.-]', '', regex=True)
                    numeric_series = pd.to_numeric(cleaned_series, errors='coerce')
                    
                    # If more than 70% can be converted to numeric, convert the column
                    if (numeric_series.notna().sum() / len(df)) > 0.7:
                        df[col] = numeric_series
                        logger.info(f"Converted column '{col}' to numeric")
                        
                except Exception as e:
                    logger.warning(f"Error cleaning column {col}: {e}")
        
        # Remove rows that are all NaN after cleaning
        df = df.dropna(how='all')
        
        # Ensure minimum data requirements
        if len(df) < 1:
            raise ValueError("DataFrame has no valid data rows")
        
        # Limit size for performance
        if len(df) > settings.max_rows:
            logger.warning(f"DataFrame too large ({len(df)} rows), taking first {settings.max_rows}")
            df = df.head(settings.max_rows)
        
        logger.info(f"Cleaned execution result: {df.shape}")
        return df
    
    def _fallback_scraping(self, url: str) -> pd.DataFrame:
        """Enhanced fallback scraping with multiple strategies"""
        logger.info("🔄 Attempting fallback scraping strategies")
        
        strategies = [
            ("pandas.read_html()[0]", lambda: pd.read_html(url)[0]),
            ("pandas.read_html()[1]", lambda: pd.read_html(url)[1]),
            ("BeautifulSoup table parsing", lambda: self._manual_table_parsing(url)),
        ]
        
        for strategy_name, strategy_func in strategies:
            try:
                logger.info(f"Trying strategy: {strategy_name}")
                df = strategy_func()
                
                # Validate result
                if isinstance(df, pd.DataFrame) and df.shape[0] > 1 and df.shape[1] > 1:
                    logger.info(f"✓ {strategy_name} successful: {df.shape}")
                    return df
                else:
                    logger.warning(f"{strategy_name} produced insufficient data: {df.shape if hasattr(df, 'shape') else 'not a dataframe'}")
                    
            except Exception as e:
                logger.warning(f"{strategy_name} failed: {e}")
                continue
        
        # Final fallback - create minimal dataframe
        logger.error("All scraping strategies failed")
        raise Exception("All scraping strategies failed")
    
    def _manual_table_parsing(self, url: str) -> pd.DataFrame:
        """Manual table parsing using BeautifulSoup"""
        logger.info("🔧 Manual table parsing")
        
        try:
            response = self.session.get(url, timeout=settings.request_timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            tables = soup.find_all('table')
            
            for i, table in enumerate(tables):
                try:
                    # Extract headers with better logic
                    headers = []
                    all_rows = table.find_all('tr')
                    
                    if not all_rows:
                        continue
                    
                    # Try to find headers from first few rows
                    for row in all_rows[:3]:
                        potential_headers = []
                        
                        # Try th elements first
                        th_elements = row.find_all('th')
                        if th_elements:
                            potential_headers = [th.get_text(strip=True) for th in th_elements]
                        else:
                            # Try td elements
                            td_elements = row.find_all('td')
                            if td_elements:
                                potential_headers = [td.get_text(strip=True) for td in td_elements]
                        
                        if potential_headers and len(potential_headers) >= 2:
                            headers = potential_headers
                            break
                    
                    if len(headers) < 2:
                        continue
                    
                    # Extract data rows
                    data_rows = []
                    data_start_idx = 1 if all_rows[0].find('th') else 0
                    
                    for row in all_rows[data_start_idx:]:
                        cells = row.find_all(['td', 'th'])
                        if len(cells) >= len(headers):
                            row_data = [cell.get_text(strip=True) for cell in cells[:len(headers)]]
                            # Only add rows with some content
                            if any(cell.strip() for cell in row_data):
                                data_rows.append(row_data)
                    
                    if len(data_rows) >= 1:  # At least 1 data row
                        df = pd.DataFrame(data_rows, columns=headers)
                        logger.info(f"Manual parsing successful for table {i}: {df.shape}")
                        return df
                        
                except Exception as e:
                    logger.warning(f"Manual parsing failed for table {i}: {e}")
                    continue
            
            raise Exception("Manual table parsing found no suitable tables")
            
        except Exception as e:
            logger.error(f"Manual table parsing error: {e}")
            raise Exception(f"Manual table parsing failed: {str(e)}")
    
    async def _process_task(self, df: pd.DataFrame, schema: dict, task: str) -> str:
        """Process individual task based on type"""
        task_lower = task.lower()
        
        # Determine task type
        if any(word in task_lower for word in ['plot', 'scatterplot', 'scatter', 'chart', 'graph']):
            return await self._create_plot(df, task)
        elif any(word in task_lower for word in ['correlation', 'corr']):
            return await self._calculate_correlation(df, task)
        elif any(word in task_lower for word in ['regression', 'slope']):
            return await self._calculate_regression(df, task)
        else:
            return await self._process_numeric_query(df, schema, task)
    
    async def _process_numeric_query(self, df: pd.DataFrame, schema: dict, task: str) -> str:
        """Process numeric queries using pandas/numpy or LLM"""
        logger.info("🔢 Processing numeric query")
        
        # Try direct pandas computation first
        try:
            # Simple counting queries
            if 'how many' in task.lower() or 'count' in task.lower():
                # Extract conditions from task
                if 'before' in task.lower() and any(char.isdigit() for char in task):
                    year_match = re.search(r'(\d{4})', task)
                    if year_match:
                        year = int(year_match.group(1))
                        # Find year column
                        year_cols = [col for col in df.columns if 'year' in col.lower()]
                        if year_cols:
                            result = len(df[df[year_cols[0]] < year])
                            return str(result)
                
                # Extract value conditions
                value_matches = re.findall(r'\$?([\d.]+)\s*(?:bn|billion|mil|million)', task.lower())
                if value_matches:
                    target_value = float(value_matches[0])
                    if 'bn' in task.lower() or 'billion' in task.lower():
                        target_value *= 1e9
                    elif 'mil' in task.lower() or 'million' in task.lower():
                        target_value *= 1e6
                    
                    # Find likely gross/revenue columns
                    value_cols = [col for col in df.columns if any(term in col.lower() 
                                for term in ['gross', 'revenue', 'worldwide', 'box', 'total'])]
                    
                    if value_cols:
                        col = value_cols[0]
                        # Clean and convert to numeric
                        df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')
                        result = len(df[df[col] >= target_value])
                        return str(result)
            
            # Earliest/oldest queries
            elif 'earliest' in task.lower() or 'oldest' in task.lower() or 'first' in task.lower():
                value_matches = re.findall(r'\$?([\d.]+)\s*(?:bn|billion)', task.lower())
                if value_matches:
                    target_value = float(value_matches[0]) * 1e9
                    
                    value_cols = [col for col in df.columns if any(term in col.lower() 
                                for term in ['gross', 'revenue', 'worldwide', 'box', 'total'])]
                    year_cols = [col for col in df.columns if 'year' in col.lower()]
                    title_cols = [col for col in df.columns if any(term in col.lower() 
                                for term in ['title', 'film', 'movie', 'name'])]
                    
                    if value_cols and year_cols and title_cols:
                        val_col, year_col, title_col = value_cols[0], year_cols[0], title_cols[0]
                        
                        # Clean numeric data
                        df[val_col] = pd.to_numeric(df[val_col].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')
                        df[year_col] = pd.to_numeric(df[year_col].astype(str).str.extract(r'(\d{4})')[0], errors='coerce')
                        
                        # Filter and find earliest
                        filtered = df[df[val_col] >= target_value]
                        if not filtered.empty:
                            earliest = filtered.loc[filtered[year_col].idxmin()]
                            return str(earliest[title_col])
            
        except Exception as e:
            logger.warning(f"Direct computation failed: {e}")
        
        # Fallback to LLM analysis
        return await self._llm_numeric_analysis(df, schema, task)
    
    async def _llm_numeric_analysis(self, df: pd.DataFrame, schema: dict, task: str) -> str:
        """Use LLM for complex numeric analysis"""
        logger.info("🤖 Using LLM for numeric analysis")
        
        # Prepare data summary for LLM
        data_summary = {
            'columns': list(df.columns),
            'shape': df.shape,
            'sample_data': df.head(3).to_dict('records'),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()}
        }
        
        prompt = f"""
You are a data analyst. Given this dataset summary and task, provide ONLY the final answer as a string.

Dataset Summary:
{json.dumps(data_summary, indent=2)}

Task: {task}

Analyze the data and provide just the answer (number, name, or value) with no explanation.
"""
        
        try:
            if settings.ai_provider.lower() == "google":
                response = ai_client.generate_content(prompt)
                return response.text.strip()
            else:
                response = ai_client.chat.completions.create(
                    model=settings.ai_model,
                    messages=[
                        {"role": "system", "content": "You are a data analyst. Provide only the final answer."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=100
                )
                return response.choices[0].message.content.strip()
                
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return "Analysis unavailable"
    
    async def _calculate_correlation(self, df: pd.DataFrame, task: str) -> str:
        """Calculate correlation between columns"""
        logger.info("📊 Calculating correlation")
        
        try:
            # Extract column names from task
            rank_cols = [col for col in df.columns if 'rank' in col.lower()]
            peak_cols = [col for col in df.columns if 'peak' in col.lower()]
            
            if rank_cols and peak_cols:
                rank_col, peak_col = rank_cols[0], peak_cols[0]
                
                # Clean and convert to numeric
                df[rank_col] = pd.to_numeric(df[rank_col].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')
                df[peak_col] = pd.to_numeric(df[peak_col].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')
                
                # Calculate correlation
                correlation = df[rank_col].corr(df[peak_col])
                return f"{correlation:.4f}"
            
            # Fallback: try to identify numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                corr = df[numeric_cols[0]].corr(df[numeric_cols[1]])
                return f"{corr:.4f}"
                
        except Exception as e:
            logger.error(f"Correlation calculation failed: {e}")
        
        return "Correlation unavailable"
    
    async def _calculate_regression(self, df: pd.DataFrame, task: str) -> str:
        """Calculate regression slope"""
        logger.info("📈 Calculating regression slope")
        
        try:
            # Find appropriate columns
            year_cols = [col for col in df.columns if 'year' in col.lower()]
            
            # Look for delay or time-related columns
            delay_cols = [col for col in df.columns if any(term in col.lower() 
                         for term in ['delay', 'days', 'duration', 'time'])]
            
            if year_cols and delay_cols:
                x_col, y_col = year_cols[0], delay_cols[0]
            else:
                # Use first two numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) >= 2:
                    x_col, y_col = numeric_cols[0], numeric_cols[1]
                else:
                    return "Insufficient numeric data"
            
            # Clean data
            df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
            df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
            
            # Remove NaN values
            clean_data = df[[x_col, y_col]].dropna()
            
            if len(clean_data) < 2:
                return "Insufficient data points"
            
            # Calculate regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(clean_data[x_col], clean_data[y_col])
            return f"{slope:.6f}"
            
        except Exception as e:
            logger.error(f"Regression calculation failed: {e}")
            return "Regression unavailable"
    
    async def _create_plot(self, df: pd.DataFrame, task: str) -> str:
        """Create matplotlib plot and return as base64"""
        logger.info("📊 Creating plot")
        
        try:
            # Find appropriate columns for plotting
            rank_cols = [col for col in df.columns if 'rank' in col.lower()]
            peak_cols = [col for col in df.columns if 'peak' in col.lower()]
            year_cols = [col for col in df.columns if 'year' in col.lower()]
            
            plt.figure(figsize=(settings.plot_width, settings.plot_height))
            
            if rank_cols and peak_cols:
                # Rank vs Peak scatterplot
                x_col, y_col = rank_cols[0], peak_cols[0]
                xlabel, ylabel = 'Rank', 'Peak'
            elif year_cols:
                # Year-based plot
                x_col = year_cols[0]
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                y_col = [col for col in numeric_cols if col != x_col][0] if len(numeric_cols) > 1 else numeric_cols[0]
                xlabel, ylabel = 'Year', y_col
            else:
                # Use first two numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) >= 2:
                    x_col, y_col = numeric_cols[0], numeric_cols[1]
                    xlabel, ylabel = x_col, y_col
                else:
                    raise Exception("Insufficient numeric columns for plotting")
            
            # Clean data
            df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
            df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
            
            # Remove NaN values
            plot_data = df[[x_col, y_col]].dropna()
            
            if len(plot_data) < 2:
                raise Exception("Insufficient data points for plotting")
            
            # Create scatterplot
            plt.scatter(plot_data[x_col], plot_data[y_col], alpha=0.6, s=50)
            
            # Add regression line
            if len(plot_data) >= 2:
                slope, intercept, r_value, p_value, std_err = stats.linregress(plot_data[x_col], plot_data[y_col])
                line_x = np.array([plot_data[x_col].min(), plot_data[x_col].max()])
                line_y = slope * line_x + intercept
                plt.plot(line_x, line_y, 'r--', alpha=0.8, linewidth=2, label=f'Regression Line')
            
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(f'{xlabel} vs {ylabel}')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Save plot
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=settings.plot_dpi, bbox_inches='tight')
            buffer.seek(0)
            
            # Check size
            image_data = buffer.read()
            if len(image_data) > settings.max_image_size:
                # Reduce quality
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=80, bbox_inches='tight', quality=70)
                buffer.seek(0)
                image_data = buffer.read()
            
            plt.close()
            
            # Encode as base64
            image_base64 = base64.b64encode(image_data).decode()
            return f"data:image/png;base64,{image_base64}"
            
        except Exception as e:
            logger.error(f"Plot creation failed: {e}")
            # Return minimal 1px image
            return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="

# Initialize agent
agent = DataAnalystAgent()

@app.post("/analyze")
async def analyze_endpoint(request: AnalysisRequest):
    """Main analysis endpoint"""
    logger.info(f"🚀 Analysis request received")
    logger.info(f"📍 URL: {request.url}")
    logger.info(f"📝 Tasks: {len(request.tasks)}")
    
    try:
        results = await agent.analyze_data(request.url, request.tasks)
        logger.info(f"✅ Analysis completed successfully")
        return JSONResponse(content=results)
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "agent": "Data Analyst Agent v2.0"}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Data Analyst Agent - Schema-Driven Analysis",
        "version": "2.0.0",
        "endpoints": {
            "/analyze": "POST - Main analysis endpoint",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation (debug mode only)"
        },
        "example_request": {
            "url": "https://example.com/data.html",
            "tasks": [
                "How many records are there?",
                "What is the correlation between column1 and column2?",
                "Create a scatterplot of column1 vs column2"
            ]
        }
    }

if __name__ == "__main__":
    logger.info("🚀 Starting Data Analyst Agent")
    logger.info(f"🌐 Server: http://{settings.api_host}:{settings.api_port}")
    logger.info(f"📚 Docs: http://{settings.api_host}:{settings.api_port}/docs")
    
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info"
    )
