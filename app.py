"""
Enterprise Website Intelligence Platform with Advanced AI Workflows

Installation:
pip install langchain langchain-openai gradio requests beautifulsoup4 pandas

For enhanced workflow capabilities (optional):
pip install langgraph

If you get import errors, the system will automatically fall back to 
sequential processing with standard AI components.
"""

import os
import gradio as gr

# Set USER_AGENT to avoid warnings
os.environ.setdefault('USER_AGENT', 'LangChain-Website-Analyzer/1.0')
import requests
from bs4 import BeautifulSoup
import re
import os
import pandas as pd
import concurrent.futures
import time
import zipfile
import hashlib
import logging
import json
import tempfile
import shutil
from functools import lru_cache
from threading import Lock
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum

# LangChain imports - with fallbacks for compatibility
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    from langchain.chat_models import ChatOpenAI

try:
    from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
    from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError:
    from langchain.prompts import ChatPromptTemplate, PromptTemplate
    from langchain.output_parsers import JsonOutputParser, StrOutputParser
    from langchain.schema import HumanMessage, SystemMessage

try:
    from langchain.memory import ConversationBufferMemory
    from langchain.cache import InMemoryCache
    from langchain.globals import set_llm_cache
except ImportError:
    # Fallback for older versions
    ConversationBufferMemory = None
    InMemoryCache = None
    set_llm_cache = lambda x: None

try:
    from langchain_core.runnables.config import RunnableConfig
except ImportError:
    RunnableConfig = dict

# LangGraph imports - with fallbacks
try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("LangGraph not available, using simplified workflow")
    StateGraph = None
    END = None
    MemorySaver = None
    LANGGRAPH_AVAILABLE = False

# Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Enable LangChain caching if available
if InMemoryCache:
    set_llm_cache(InMemoryCache())

# Data Models
class BusinessType(Enum):
    B2B = "Business to Business"
    B2C = "Business to Consumer"
    BOTH = "Both"

class OperationType(Enum):
    MANUFACTURING = "Manufacturing"
    TRADING = "Trading"
    SERVICES = "Services"

@dataclass
class WebsiteData:
    url: str
    text: str
    soup: Optional[BeautifulSoup] = None
    html_content: Optional[str] = None
    meta_info: Dict[str, str] = None
    
@dataclass
class BusinessAnalysis:
    business_nature: str
    customer_type: str
    operation_type: str
    countries: str
    products_services: str
    target_customers: str
    target_industries: str
    business_model: str
    global_presence: str

@dataclass
class AnalysisResult:
    website: str
    status: str
    company_description: str
    business_analysis: BusinessAnalysis
    decision: str
    keyword_matches: List[str]
    processing_time: float
    error_message: str = ""

# LangGraph State
class AnalysisState:
    def __init__(self):
        self.website_data: Optional[WebsiteData] = None
        self.translated_text: str = ""
        self.company_description: str = ""
        self.business_analysis: Optional[BusinessAnalysis] = None
        self.keyword_matches: List[str] = []
        self.decision: str = ""
        self.error_message: str = ""
        self.processing_time: float = 0.0

# LangChain Components
class AdvancedWebsiteAnalyzer:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
        self.llm = ChatOpenAI(
            api_key=api_key,
            model=model,
            temperature=0.3,
            max_tokens=1500,
            request_timeout=30
        )
        
        # Initialize memory and cache
        self.memory = ConversationBufferMemory(return_messages=True)
        self.cache = {}
        self.session_locks = {}
        
        # Initialize prompts
        self._setup_prompts()
        
        # Initialize chains
        self._setup_chains()
        
        # Initialize graph
        self._setup_workflow()
    
    def _setup_prompts(self):
        """Setup LangChain prompt templates"""
        
        # Translation prompt
        self.translation_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a professional translator. Translate text to English while preserving formatting and business context."),
            ("human", "Translate the following text to English:\n\n{text}")
        ])
        
        # Company description prompt
        self.description_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a business analyst. Based on website content, provide a concise 15-word description of what this company does. 
            If insufficient information, state 'Insufficient information available to determine company purpose.'"""),
            ("human", "Website: {url}\n\nContent: {content}\n\nProvide a concise description:")
        ])
        
        # Business analysis prompt
        self.business_analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a business intelligence analyst. Analyze the website content and provide structured business information.
            
            Respond with ONLY a JSON object containing these exact fields:
            {{
                "business_nature": "Detailed description of what the company does",
                "customer_type": "Business to Business" OR "Business to Consumer" OR "Both",
                "operation_type": "Manufacturing" OR "Trading" OR "Services" OR combinations like "Manufacturing, Trading",
                "countries": "List of countries/regions mentioned or 'Not available'",
                "products_services": "Detailed list of products/services offered",
                "target_customers": "Description of their target customer base",
                "target_industries": "Industries they serve",
                "business_model": "How they generate revenue",
                "global_presence": "International operations and presence"
            }}
            
            Base your analysis only on the provided content. Be specific and detailed."""),
            ("human", "Website: {url}\n\nContent: {content}\n\nAnalyze this business:")
        ])
        
        # Detailed analysis prompt for additional pages
        self.detailed_analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a business intelligence expert. Analyze multiple pages from a company website and provide comprehensive insights.
            
            Focus on extracting:
            1. Specific products and services
            2. Target market segments
            3. Business model details
            4. Global operations
            5. Industry expertise
            
            Provide detailed, factual information based on the content."""),
            ("human", """Main Website: {main_url}
            
            Content from multiple pages:
            {combined_content}
            
            Provide detailed analysis for:
            1. Products & Services:
            2. Target Customers:
            3. Target Industries:
            4. Business Model:
            5. Global Presence:""")
        ])
    
    def _setup_chains(self):
        """Setup LangChain processing chains"""
        
        # Translation chain
        self.translation_chain = (
            self.translation_prompt 
            | self.llm 
            | StrOutputParser()
        )
        
        # Description chain
        self.description_chain = (
            self.description_prompt
            | self.llm
            | StrOutputParser()
        )
        
        # Business analysis chain with error handling
        try:
            self.business_analysis_chain = (
                self.business_analysis_prompt
                | self.llm
                | JsonOutputParser()
            )
        except Exception as e:
            logger.warning(f"JSON parser setup failed: {e}, using string parser")
            self.business_analysis_chain = (
                self.business_analysis_prompt
                | self.llm
                | StrOutputParser()
            )
        
        # Detailed analysis chain
        self.detailed_analysis_chain = (
            self.detailed_analysis_prompt
            | self.llm
            | StrOutputParser()
        )
    
    def _setup_workflow(self):
        """Setup LangGraph workflow with fallback"""
        
        if not LANGGRAPH_AVAILABLE:
            logger.info("Using simplified sequential workflow (LangGraph not available)")
            self.graph = None
            return
        
        # Define the graph
        workflow = StateGraph(dict)
        
        # Add nodes
        workflow.add_node("extract_content", self._extract_content_node)
        workflow.add_node("translate_text", self._translate_node)
        workflow.add_node("generate_description", self._description_node)
        workflow.add_node("analyze_business", self._business_analysis_node)
        workflow.add_node("detailed_analysis", self._detailed_analysis_node)
        workflow.add_node("make_decision", self._decision_node)
        workflow.add_node("finalize_result", self._finalize_node)
        
        # Define edges
        workflow.set_entry_point("extract_content")
        workflow.add_edge("extract_content", "translate_text")
        workflow.add_edge("translate_text", "generate_description")
        workflow.add_edge("generate_description", "analyze_business")
        workflow.add_edge("analyze_business", "detailed_analysis")
        workflow.add_edge("detailed_analysis", "make_decision")
        workflow.add_edge("make_decision", "finalize_result")
        workflow.add_edge("finalize_result", END)
        
        # Compile the graph
        try:
            self.graph = workflow.compile(checkpointer=MemorySaver())
        except Exception as e:
            logger.warning(f"Failed to compile LangGraph workflow: {e}, using fallback")
            self.graph = None
    
    def _extract_content_node(self, state: dict) -> dict:
        """Extract content from website"""
        try:
            url = state["url"]
            start_time = time.time()
            
            # Use existing extraction logic
            extracted_text, soup, html_content = self._extract_text_from_url(url)
            
            if extracted_text.startswith("Error"):
                state["error_message"] = extracted_text
                state["status"] = "Error"
                return state
            
            # Extract meta information
            meta_info = self._extract_meta_info(soup) if soup else {}
            
            state["website_data"] = {
                "url": url,
                "text": extracted_text,
                "soup": soup,
                "html_content": html_content,
                "meta_info": meta_info
            }
            state["processing_time"] = time.time() - start_time
            
        except Exception as e:
            logger.error(f"Content extraction error: {str(e)}")
            state["error_message"] = f"Content extraction error: {str(e)}"
            state["status"] = "Error"
        
        return state
    
    def _translate_node(self, state: dict) -> dict:
        """Translate content if needed"""
        try:
            website_data = state.get("website_data")
            if not website_data or state.get("error_message"):
                return state
            
            text = website_data["text"]
            
            # Check if translation is needed
            if self._is_likely_english(text):
                state["translated_text"] = text
                state["translation_status"] = "No translation needed"
            else:
                # Use LangChain for translation
                translated = self.translation_chain.invoke({"text": text[:2000]})
                state["translated_text"] = translated
                state["translation_status"] = "Translation successful"
                
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            state["translated_text"] = website_data["text"] if website_data else ""
            state["translation_status"] = f"Translation error: {str(e)}"
        
        return state
    
    def _description_node(self, state: dict) -> dict:
        """Generate company description"""
        try:
            if state.get("error_message"):
                return state
            
            website_data = state.get("website_data")
            translated_text = state.get("translated_text", "")
            
            if not translated_text:
                state["company_description"] = "Unable to generate description"
                return state
            
            # Use LangChain for description generation
            description = self.description_chain.invoke({
                "url": website_data["url"],
                "content": translated_text[:1000]
            })
            
            state["company_description"] = description.strip()
            
        except Exception as e:
            logger.error(f"Description generation error: {str(e)}")
            state["company_description"] = f"Error generating description: {str(e)}"
        
        return state
    
    def _business_analysis_node(self, state: dict) -> dict:
        """Analyze business using LangChain"""
        try:
            if state.get("error_message"):
                return state
            
            website_data = state.get("website_data")
            translated_text = state.get("translated_text", "")
            
            if not translated_text:
                state["business_analysis"] = self._get_default_analysis()
                return state
            
            # Use LangChain for business analysis
            analysis_result = self.business_analysis_chain.invoke({
                "url": website_data["url"],
                "content": translated_text[:2500]
            })
            
            # Handle both JSON and string responses
            if isinstance(analysis_result, dict):
                # Direct JSON response
                business_analysis = BusinessAnalysis(
                    business_nature=analysis_result.get("business_nature", "Unable to determine"),
                    customer_type=analysis_result.get("customer_type", "Unable to determine"),
                    operation_type=analysis_result.get("operation_type", "Unable to determine"),
                    countries=analysis_result.get("countries", "Not available"),
                    products_services=analysis_result.get("products_services", "Not available"),
                    target_customers=analysis_result.get("target_customers", "Not available"),
                    target_industries=analysis_result.get("target_industries", "Not available"),
                    business_model=analysis_result.get("business_model", "Not available"),
                    global_presence=analysis_result.get("global_presence", "Not available")
                )
            else:
                # String response - parse manually
                business_analysis = self._parse_string_analysis(str(analysis_result))
            
            state["business_analysis"] = business_analysis
            
        except Exception as e:
            logger.error(f"Business analysis error: {str(e)}")
            state["business_analysis"] = self._get_default_analysis()
        
        return state
    
    def _parse_string_analysis(self, analysis_text: str) -> BusinessAnalysis:
        """Parse string analysis response to BusinessAnalysis object"""
        try:
            # Try to extract JSON from string
            import json
            json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
            if json_match:
                analysis_dict = json.loads(json_match.group())
                return BusinessAnalysis(
                    business_nature=analysis_dict.get("business_nature", "Unable to determine"),
                    customer_type=analysis_dict.get("customer_type", "Unable to determine"),
                    operation_type=analysis_dict.get("operation_type", "Unable to determine"),
                    countries=analysis_dict.get("countries", "Not available"),
                    products_services=analysis_dict.get("products_services", "Not available"),
                    target_customers=analysis_dict.get("target_customers", "Not available"),
                    target_industries=analysis_dict.get("target_industries", "Not available"),
                    business_model=analysis_dict.get("business_model", "Not available"),
                    global_presence=analysis_dict.get("global_presence", "Not available")
                )
        except Exception as e:
            logger.warning(f"Failed to parse JSON from string: {e}")
        
        # Fallback: extract from text patterns
        return BusinessAnalysis(
            business_nature=self._extract_field_from_text(analysis_text, "business_nature") or "Unable to determine",
            customer_type=self._extract_field_from_text(analysis_text, "customer_type") or "Unable to determine",
            operation_type=self._extract_field_from_text(analysis_text, "operation_type") or "Unable to determine",
            countries=self._extract_field_from_text(analysis_text, "countries") or "Not available",
            products_services=self._extract_field_from_text(analysis_text, "products_services") or "Not available",
            target_customers=self._extract_field_from_text(analysis_text, "target_customers") or "Not available",
            target_industries=self._extract_field_from_text(analysis_text, "target_industries") or "Not available",
            business_model=self._extract_field_from_text(analysis_text, "business_model") or "Not available",
            global_presence=self._extract_field_from_text(analysis_text, "global_presence") or "Not available"
        )
    
    def _extract_field_from_text(self, text: str, field_name: str) -> Optional[str]:
        """Extract field value from text using patterns"""
        patterns = [
            rf'"{field_name}":\s*"([^"]*)"',
            rf'{field_name}:\s*"([^"]*)"',
            rf'{field_name}:\s*([^\n,}}]*)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _detailed_analysis_node(self, state: dict) -> dict:
        """Perform detailed analysis of additional pages"""
        try:
            if state.get("error_message"):
                return state
            
            website_data = state.get("website_data")
            translated_text = state.get("translated_text", "")
            
            # Extract additional pages
            soup = website_data.get("soup")
            if soup:
                about_links, product_links = self._extract_key_page_links(soup, website_data["url"])
                
                additional_content = ""
                for link, text in (about_links + product_links)[:2]:  # Limit to 2 additional pages
                    try:
                        page_text, _, _ = self._extract_text_from_url(link)
                        if not page_text.startswith("Error"):
                            additional_content += f"\n\nPage: {link}\nContent: {page_text[:1000]}"
                    except Exception as e:
                        logger.warning(f"Error fetching additional page {link}: {str(e)}")
                
                if additional_content:
                    # Use detailed analysis chain
                    detailed_result = self.detailed_analysis_chain.invoke({
                        "main_url": website_data["url"],
                        "combined_content": translated_text[:1500] + additional_content
                    })
                    
                    # Update business analysis with detailed information
                    business_analysis = state.get("business_analysis")
                    if business_analysis and detailed_result:
                        # Parse detailed result and update fields
                        self._update_analysis_with_details(business_analysis, detailed_result)
                        state["business_analysis"] = business_analysis
            
        except Exception as e:
            logger.error(f"Detailed analysis error: {str(e)}")
        
        return state
    
    def _decision_node(self, state: dict) -> dict:
        """Make final decision based on criteria"""
        try:
            if state.get("error_message"):
                state["decision"] = "Error"
                return state
            
            business_analysis = state.get("business_analysis")
            translated_text = state.get("translated_text", "")
            business_terms = state.get("business_terms", [])
            customer_types = state.get("customer_types", [])
            operation_types = state.get("operation_types", [])
            
            # Check keyword matches
            keyword_matches = []
            text_lower = translated_text.lower()
            for term in business_terms:
                if term.lower() in text_lower:
                    keyword_matches.append(term)
            
            state["keyword_matches"] = keyword_matches
            
            # Apply decision logic
            customer_match = self._check_customer_type_match(business_analysis.customer_type, customer_types)
            operation_match = self._check_operation_type_match(business_analysis.operation_type, operation_types)
            terms_match = len(keyword_matches) > 0 if business_terms else True
            
            if customer_match and operation_match and terms_match:
                state["decision"] = "Accepted"
            else:
                rejection_reasons = []
                if not customer_match:
                    rejection_reasons.append("Customer Type Mismatch")
                if not operation_match:
                    rejection_reasons.append("Operation Type Mismatch")
                if not terms_match:
                    rejection_reasons.append("No Business Terms Found")
                
                state["decision"] = f"Rejected ({', '.join(rejection_reasons)})"
            
        except Exception as e:
            logger.error(f"Decision error: {str(e)}")
            state["decision"] = "Error in decision making"
        
        return state
    
    def _finalize_node(self, state: dict) -> dict:
        """Finalize the analysis result"""
        try:
            website_data = state.get("website_data", {})
            business_analysis = state.get("business_analysis")
            
            # Create final result
            result = AnalysisResult(
                website=website_data.get("url", "Unknown"),
                status="Success" if not state.get("error_message") else "Error",
                company_description=state.get("company_description", "N/A"),
                business_analysis=business_analysis or self._get_default_analysis(),
                decision=state.get("decision", "Unknown"),
                keyword_matches=state.get("keyword_matches", []),
                processing_time=state.get("processing_time", 0.0),
                error_message=state.get("error_message", "")
            )
            
            state["final_result"] = result
            
        except Exception as e:
            logger.error(f"Finalization error: {str(e)}")
            state["error_message"] = f"Finalization error: {str(e)}"
        
        return state
    
    def analyze_website(self, url: str, business_terms: List[str] = None, 
                       customer_types: List[str] = None, 
                       operation_types: List[str] = None) -> AnalysisResult:
        """Analyze a single website using LangGraph workflow or fallback"""
        
        # Prepare initial state
        initial_state = {
            "url": url,
            "business_terms": business_terms or [],
            "customer_types": customer_types or [],
            "operation_types": operation_types or []
        }
        
        try:
            if self.graph and LANGGRAPH_AVAILABLE:
                # Use LangGraph workflow
                config = {"configurable": {"thread_id": f"analysis_{hashlib.md5(url.encode()).hexdigest()}"}}
                final_state = self.graph.invoke(initial_state, config)
                return final_state.get("final_result")
            else:
                # Use sequential fallback processing
                return self._sequential_analysis(initial_state)
                
        except Exception as e:
            logger.error(f"Analysis error for {url}: {str(e)}")
            return AnalysisResult(
                website=url,
                status="Error",
                company_description="Analysis execution error",
                business_analysis=self._get_default_analysis(),
                decision="Error",
                keyword_matches=[],
                processing_time=0.0,
                error_message=str(e)
            )
    
    def _sequential_analysis(self, initial_state: dict) -> AnalysisResult:
        """Fallback sequential analysis when LangGraph is not available"""
        try:
            start_time = time.time()
            
            # Sequential processing
            state = initial_state.copy()
            state = self._extract_content_node(state)
            if state.get("error_message"):
                return self._create_error_result(state, time.time() - start_time)
            
            state = self._translate_node(state)
            state = self._description_node(state)
            state = self._business_analysis_node(state)
            state = self._detailed_analysis_node(state)
            state = self._decision_node(state)
            state = self._finalize_node(state)
            
            return state.get("final_result")
            
        except Exception as e:
            logger.error(f"Sequential analysis error: {str(e)}")
            return AnalysisResult(
                website=initial_state.get("url", "Unknown"),
                status="Error",
                company_description="Sequential analysis error",
                business_analysis=self._get_default_analysis(),
                decision="Error",
                keyword_matches=[],
                processing_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def _create_error_result(self, state: dict, processing_time: float) -> AnalysisResult:
        """Create error result from state"""
        return AnalysisResult(
            website=state.get("url", "Unknown"),
            status="Error",
            company_description="Analysis error",
            business_analysis=self._get_default_analysis(),
            decision="Error",
            keyword_matches=[],
            processing_time=processing_time,
            error_message=state.get("error_message", "Unknown error")
        )
    
    # Helper methods (keeping original functionality)
    def _extract_text_from_url(self, url: str) -> Tuple[str, Optional[BeautifulSoup], Optional[str]]:
        """Extract text from URL (keeping original logic)"""
        try:
            formatted_url = self._format_url(url)
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/html,application/xhtml+xml',
                'Accept-Language': 'en-US,en;q=0.9',
                'Connection': 'keep-alive',
                'Cache-Control': 'max-age=0'
            }
            
            response = requests.get(formatted_url, headers=headers, timeout=10, verify=True)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            for element in soup(["script", "style", "nav", "footer", "iframe", "header"]):
                element.extract()
            
            text = soup.get_text(separator='\n')
            lines = (line.strip() for line in text.splitlines())
            text = '\n'.join(line for line in lines if line)
            
            return text, soup, response.text
            
        except Exception as e:
            logger.error(f"Error extracting text from {url}: {str(e)}")
            return f"Error: {str(e)}", None, None
    
    def _format_url(self, url: str) -> str:
        """Format URL properly"""
        url = url.strip()
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        if not url.endswith('/'):
            url = url + '/'
        return url
    
    def _is_likely_english(self, text: str, sample_size: int = 500) -> bool:
        """Check if text is likely in English"""
        common_words = set(['the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'I', 'it', 'for', 
                           'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by', 
                           'from', 'they', 'we'])
        sample = text[:sample_size].lower()
        words = re.findall(r'\b\w+\b', sample)
        if not words:
            return True
        english_count = sum(1 for word in words if word in common_words)
        return (english_count / len(words) > 0.15) if words else True
    
    def _extract_meta_info(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract meta information from soup"""
        meta_info = {}
        if not soup:
            return meta_info
            
        if soup.title:
            meta_info['title'] = soup.title.get_text()
        
        for meta in soup.find_all('meta'):
            name = meta.get('name', '').lower()
            content = meta.get('content', '')
            if name in ['description', 'keywords'] and content:
                meta_info[name] = content
                
        return meta_info
    
    def _extract_key_page_links(self, soup: BeautifulSoup, base_url: str, max_links: int = 3) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """Extract about and product page links"""
        if not soup:
            return [], []
            
        about_links = []
        product_links = []
        
        about_terms = ['about', 'company', 'who we are', 'about us', 'our story', 'our business']
        product_terms = ['product', 'service', 'solution', 'offering', 'what we do', 'our work']
        
        for link in soup.find_all('a', href=True):
            href = link.get('href', '').lower()
            link_text = link.get_text().lower().strip()
            
            if not link_text and not href:
                continue
                
            if any(skip in href for skip in ['twitter.com', 'facebook.com', 'linkedin.com', 'instagram.com', 
                                            'youtube.com', 'login', 'signin', 'contact', 'privacy', 'terms']):
                continue
            
            formatted_href = href
            if href.startswith('/'):
                formatted_href = base_url + href.lstrip('/')
            elif not href.startswith(('http://', 'https://')):
                formatted_href = base_url + href
                
            about_score = sum(5 if term in href else 2 if term in link_text else 0 for term in about_terms)
            product_score = sum(5 if term in href else 2 if term in link_text else 0 for term in product_terms)
            
            if about_score > 0:
                about_links.append((formatted_href, link_text, about_score))
            elif product_score > 0:
                product_links.append((formatted_href, link_text, product_score))
        
        about_links.sort(key=lambda x: x[2], reverse=True)
        product_links.sort(key=lambda x: x[2], reverse=True)
        
        return [(link, text) for link, text, _ in about_links[:max_links]], [(link, text) for link, text, _ in product_links[:max_links]]
    
    def _get_default_analysis(self) -> BusinessAnalysis:
        """Get default business analysis"""
        return BusinessAnalysis(
            business_nature="Unable to determine",
            customer_type="Unable to determine",
            operation_type="Unable to determine",
            countries="Not available",
            products_services="Not available",
            target_customers="Not available",
            target_industries="Not available",
            business_model="Not available",
            global_presence="Not available"
        )
    
    def _update_analysis_with_details(self, analysis: BusinessAnalysis, detailed_result: str):
        """Update analysis with detailed information"""
        # Parse the detailed result and update relevant fields
        # This is a simplified implementation - you could make it more sophisticated
        if "Products & Services:" in detailed_result:
            products_section = detailed_result.split("Products & Services:")[1].split("\n")[0:3]
            if products_section:
                analysis.products_services = " ".join(products_section).strip()
    
    def _check_customer_type_match(self, business_customer_type: str, filter_customer_types: List[str]) -> bool:
        """Check if customer type matches filter"""
        if not filter_customer_types:
            return True
        
        if business_customer_type == "Unable to determine":
            return False
        
        if business_customer_type == "Both":
            return any(ct in ["Both", "Business to Business", "Business to Consumer"] for ct in filter_customer_types)
        
        return business_customer_type in filter_customer_types
    
    def _check_operation_type_match(self, business_operation_type: str, filter_operation_types: List[str]) -> bool:
        """Check if operation type matches filter"""
        if not filter_operation_types:
            return True
        
        if business_operation_type == "Unable to determine":
            return False
        
        business_operation_types = [op.strip() for op in business_operation_type.split(',')]
        return any(op == business_op for op in filter_operation_types for business_op in business_operation_types)

# Enhanced Streaming Processor - FIXED VERSION
class StreamingWebsiteProcessor:
    def __init__(self, api_key: str, max_workers: int = 4):
        self.analyzer = AdvancedWebsiteAnalyzer(api_key)
        self.max_workers = max_workers
        self.results = []
        
    def process_websites_stream(self, urls: List[str], business_terms: List[str] = None,
                               customer_types: List[str] = None, 
                               operation_types: List[str] = None):
        """Process websites with streaming results using ThreadPoolExecutor - SIMPLIFIED VERSION"""
        
        # FIX: Clear previous results to avoid accumulation across runs
        self.results = []
        
        total_urls = len(urls)
        
        def analyze_single_website(url: str) -> AnalysisResult:
            return self.analyzer.analyze_website(url, business_terms, customer_types, operation_types)
        
        # Process websites concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_url = {executor.submit(analyze_single_website, url): url for url in urls}
            
            completed = 0
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                completed += 1
                
                try:
                    result = future.result()
                    self.results.append(result)
                    
                    # Yield progress update
                    progress = int((completed / total_urls) * 100)
                    status = f"🔄 PROCESSING ({completed}/{total_urls}) - {progress}% Complete"
                    
                    # FIX: Don't yield DataFrames here, just status and log
                    yield status, None, None, f"Completed: {url}", False
                    
                except Exception as e:
                    logger.error(f"Error processing {url}: {str(e)}")
                    error_result = AnalysisResult(
                        website=url,
                        status="Error",
                        company_description="Processing error",
                        business_analysis=self.analyzer._get_default_analysis(),
                        decision="Error",
                        keyword_matches=[],
                        processing_time=0.0,
                        error_message=str(e)
                    )
                    self.results.append(error_result)
                    
                    yield f"❌ ERROR ({completed}/{total_urls})", None, None, f"Error: {url}", False
        
        # Final completion signal
        accepted = sum(1 for r in self.results if r.decision == "Accepted")
        rejected = sum(1 for r in self.results if r.decision.startswith("Rejected"))
        errors = sum(1 for r in self.results if r.status == "Error")
        
        final_status = f"🎉 Analysis Complete! {accepted} Accepted, {rejected} Rejected, {errors} Errors"
        
        # Generate CSV
        csv_file = self._save_results_to_csv()
        
        # FIX: Signal completion
        yield final_status, None, csv_file, "Analysis completed successfully!", True
    
    def _result_to_dataframe_row(self, result: AnalysisResult) -> pd.DataFrame:
        """Convert analysis result to DataFrame row"""
        row_data = {
            "Website": result.website,
            "Decision": result.decision,
            "Description": result.company_description.replace('\n', ' ').replace('\r', ''),
            "Business": result.business_analysis.business_nature.replace('\n', ' ').replace('\r', ''),
            "Customers": result.business_analysis.customer_type,
            "Operations": result.business_analysis.operation_type,
            "Countries": result.business_analysis.countries.replace('\n', ' ').replace('\r', ''),
            "Products": result.business_analysis.products_services.replace('\n', ' ').replace('\r', ''),
            "Industries": result.business_analysis.target_industries.replace('\n', ' ').replace('\r', ''),
            "Model": result.business_analysis.business_model.replace('\n', ' ').replace('\r', ''),
            "Global": result.business_analysis.global_presence.replace('\n', ' ').replace('\r', ''),
            "Keywords": ", ".join(result.keyword_matches) if result.keyword_matches else "None",
            "Status": result.status,
            "Time": f"{result.processing_time:.1f}s"
        }
        return pd.DataFrame([row_data])
    
    def _results_to_dataframe(self) -> pd.DataFrame:
        """Convert all results to DataFrame"""
        if not self.results:
            return pd.DataFrame()
        
        rows = []
        for result in self.results:
            row_data = {
                "Website": result.website,
                "Decision": result.decision,
                "Description": result.company_description.replace('\n', ' ').replace('\r', ''),
                "Business": result.business_analysis.business_nature.replace('\n', ' ').replace('\r', ''),
                "Customers": result.business_analysis.customer_type,
                "Operations": result.business_analysis.operation_type,
                "Countries": result.business_analysis.countries.replace('\n', ' ').replace('\r', ''),
                "Products": result.business_analysis.products_services.replace('\n', ' ').replace('\r', ''),
                "Industries": result.business_analysis.target_industries.replace('\n', ' ').replace('\r', ''),
                "Model": result.business_analysis.business_model.replace('\n', ' ').replace('\r', ''),
                "Global": result.business_analysis.global_presence.replace('\n', ' ').replace('\r', ''),
                "Keywords": ", ".join(result.keyword_matches) if result.keyword_matches else "None",
                "Status": result.status,
                "Time": f"{result.processing_time:.1f}s"
            }
            rows.append(row_data)
        
        return pd.DataFrame(rows)
    
    def _save_results_to_csv(self) -> str:
        """Save results to CSV file"""
        try:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            temp_dir = tempfile.mkdtemp()
            csv_filename = os.path.join(temp_dir, f"enterprise_analysis_{timestamp}.csv")
            
            df = self._results_to_dataframe()
            df.to_csv(csv_filename, index=False, escapechar='\\', quoting=1)
            
            return csv_filename if os.path.exists(csv_filename) else None
        except Exception as e:
            logger.error(f"Error saving CSV: {str(e)}")
            return None
    
    def _save_detailed_reports(self) -> str:
        """Generate and save detailed reports as ZIP file"""
        try:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            temp_dir = tempfile.mkdtemp()
            zip_filename = os.path.join(temp_dir, f"detailed_reports_{timestamp}.zip")
            
            with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Generate individual reports for each website
                for i, result in enumerate(self.results):
                    report_content = self._generate_individual_report(result)
                    report_name = f"report_{i+1}_{result.website.replace('/', '_').replace(':', '_')}.txt"
                    zipf.writestr(report_name, report_content)
                
                # Generate summary report
                summary_content = self._generate_summary_report()
                zipf.writestr("summary_report.txt", summary_content)
                
                # Add CSV to the zip
                df = self._results_to_dataframe()
                csv_content = df.to_csv(index=False)
                zipf.writestr("analysis_results.csv", csv_content)
            
            return zip_filename if os.path.exists(zip_filename) else None
        except Exception as e:
            logger.error(f"Error generating reports: {str(e)}")
            return None
    
    def _generate_individual_report(self, result: AnalysisResult) -> str:
        """Generate detailed report for individual website"""
        report = f"""
WEBSITE ANALYSIS REPORT
======================

Website: {result.website}
Status: {result.status}
Decision: {result.decision}
Processing Time: {result.processing_time:.2f} seconds

COMPANY DESCRIPTION
==================
{result.company_description}

BUSINESS ANALYSIS
================
Business Nature: {result.business_analysis.business_nature}
Customer Type: {result.business_analysis.customer_type}
Operation Type: {result.business_analysis.operation_type}
Countries/Regions: {result.business_analysis.countries}
Products & Services: {result.business_analysis.products_services}
Target Customers: {result.business_analysis.target_customers}
Target Industries: {result.business_analysis.target_industries}
Business Model: {result.business_analysis.business_model}
Global Presence: {result.business_analysis.global_presence}

KEYWORD MATCHES
==============
{', '.join(result.keyword_matches) if result.keyword_matches else 'No keywords matched'}

ERROR INFORMATION
================
{result.error_message if result.error_message else 'No errors reported'}

Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
        return report
    
    def _generate_summary_report(self) -> str:
        """Generate summary report for all analyzed websites"""
        total = len(self.results)
        accepted = sum(1 for r in self.results if r.decision == "Accepted")
        rejected = sum(1 for r in self.results if r.decision.startswith("Rejected"))
        errors = sum(1 for r in self.results if r.status == "Error")
        
        avg_time = sum(r.processing_time for r in self.results) / total if total > 0 else 0
        
        # Analysis by customer type
        customer_types = {}
        for result in self.results:
            ct = result.business_analysis.customer_type
            customer_types[ct] = customer_types.get(ct, 0) + 1
        
        # Analysis by operation type
        operation_types = {}
        for result in self.results:
            ot = result.business_analysis.operation_type
            operation_types[ot] = operation_types.get(ot, 0) + 1
        
        summary = f"""
ENTERPRISE WEBSITE INTELLIGENCE SUMMARY REPORT
==============================================

OVERVIEW
========
Total Websites Analyzed: {total}
Accepted: {accepted} ({accepted/total*100:.1f}%)
Rejected: {rejected} ({rejected/total*100:.1f}%)
Errors: {errors} ({errors/total*100:.1f}%)
Average Processing Time: {avg_time:.2f} seconds

CUSTOMER TYPE DISTRIBUTION
=========================
"""
        for ct, count in customer_types.items():
            summary += f"{ct}: {count} ({count/total*100:.1f}%)\n"
        
        summary += f"""
OPERATION TYPE DISTRIBUTION
===========================
"""
        for ot, count in operation_types.items():
            summary += f"{ot}: {count} ({count/total*100:.1f}%)\n"
        
        summary += f"""
ACCEPTED WEBSITES
================
"""
        for result in self.results:
            if result.decision == "Accepted":
                summary += f"- {result.website}: {result.company_description}\n"
        
        summary += f"""
REJECTED WEBSITES
================
"""
        for result in self.results:
            if result.decision.startswith("Rejected"):
                summary += f"- {result.website}: {result.decision}\n"
        
        summary += f"""
Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
        return summary
    
    def _save_raw_data(self) -> str:
        """Save raw analysis data as JSON"""
        try:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            temp_dir = tempfile.mkdtemp()
            json_filename = os.path.join(temp_dir, f"raw_analysis_data_{timestamp}.json")
            
            # Convert results to serializable format
            raw_data = []
            for result in self.results:
                raw_data.append({
                    "website": result.website,
                    "status": result.status,
                    "company_description": result.company_description,
                    "business_analysis": asdict(result.business_analysis),
                    "decision": result.decision,
                    "keyword_matches": result.keyword_matches,
                    "processing_time": result.processing_time,
                    "error_message": result.error_message,
                    "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
                })
            
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(raw_data, f, indent=2, ensure_ascii=False)
            
            return json_filename if os.path.exists(json_filename) else None
        except Exception as e:
            logger.error(f"Error saving raw data: {str(e)}")
            return None

# Enhanced Gradio Interface - FIXED VERSION
def create_enterprise_ui():
    """Create the sophisticated enterprise interface with enhanced styling"""
    
    processor = None
    
    with gr.Blocks(
        title="Enterprise Website Intelligence Platform",
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="slate",
            neutral_hue="stone",
            font=gr.themes.GoogleFont("Poppins")
        ),
        css="""
        /* Elegant Enterprise Design */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');
        
        body, .gradio-container {
            background: linear-gradient(145deg, #fafbfc 0%, #f1f3f4 50%, #e8eaed 100%) !important;
            min-height: 100vh;
            font-family: 'Poppins', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        .enterprise-header {
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 50%, #f1f5f9 100%);
            border: 1px solid rgba(99, 102, 241, 0.1);
            border-radius: 20px;
            padding: 3rem 2.5rem;
            margin-bottom: 2.5rem;
            text-align: center;
            box-shadow: 
                0 10px 35px rgba(99, 102, 241, 0.08),
                0 1px 3px rgba(0, 0, 0, 0.05);
            position: relative;
            overflow: hidden;
        }
        
        .enterprise-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 25%, #a855f7 50%, #8b5cf6 75%, #6366f1 100%);
        }
        
        .enterprise-header::after {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(99, 102, 241, 0.03) 0%, transparent 70%);
            animation: glow 8s ease-in-out infinite;
        }
        
        @keyframes glow {
            0%, 100% { transform: rotate(0deg) scale(1); }
            50% { transform: rotate(180deg) scale(1.1); }
        }
        
        .enterprise-title {
            font-size: 3.2rem;
            font-weight: 700;
            color: #1e293b;
            margin-bottom: 1rem;
            letter-spacing: -1.5px;
            background: linear-gradient(135deg, #1e293b 0%, #475569 30%, #6366f1 70%, #8b5cf6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            position: relative;
            z-index: 2;
        }
        
        .enterprise-subtitle {
            font-size: 1.2rem;
            color: #64748b;
            font-weight: 400;
            line-height: 1.7;
            max-width: 750px;
            margin: 0 auto;
            position: relative;
            z-index: 2;
        }
        
        /* Sophisticated Group Styling */
        .gradio-group {
            background: linear-gradient(135deg, #ffffff 0%, #fefefe 100%) !important;
            border: 1px solid rgba(99, 102, 241, 0.08) !important;
            border-radius: 16px !important;
            padding: 2rem !important;
            margin-bottom: 2rem !important;
            box-shadow: 
                0 4px 20px rgba(99, 102, 241, 0.06),
                0 1px 3px rgba(0, 0, 0, 0.03) !important;
            transition: all 0.3s ease !important;
        }
        
        .gradio-group:hover {
            transform: translateY(-2px) !important;
            box-shadow: 
                0 8px 30px rgba(99, 102, 241, 0.1),
                0 2px 6px rgba(0, 0, 0, 0.05) !important;
        }
        
        /* Elegant Section Headers */
        .gradio-group h3 {
            color: #1e293b !important;
            font-weight: 600 !important;
            font-size: 1.4rem !important;
            margin-bottom: 1.5rem !important;
            padding-bottom: 0.75rem !important;
            border-bottom: 2px solid #f1f5f9 !important;
            background: linear-gradient(90deg, #6366f1, #8b5cf6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        /* Premium Input Styling */
        .gradio-textbox, .gradio-dropdown {
            border: 1.5px solid #e2e8f0 !important;
            border-radius: 12px !important;
            background: linear-gradient(135deg, #ffffff 0%, #fafbfc 100%) !important;
            transition: all 0.3s ease !important;
            font-family: 'Inter', sans-serif !important;
            padding: 0.875rem !important;
        }
        
        .gradio-textbox:focus, .gradio-dropdown:focus {
            border-color: #6366f1 !important;
            box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.1) !important;
            background: #ffffff !important;
            transform: translateY(-1px) !important;
        }
        
        /* Sophisticated Button Styling */
        .gradio-button {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
            border: none !important;
            border-radius: 12px !important;
            color: white !important;
            font-weight: 600 !important;
            font-family: 'Poppins', sans-serif !important;
            padding: 1rem 2rem !important;
            transition: all 0.3s ease !important;
            box-shadow: 
                0 4px 15px rgba(99, 102, 241, 0.25),
                0 1px 3px rgba(0, 0, 0, 0.1) !important;
            position: relative !important;
            overflow: hidden !important;
        }
        
        .gradio-button::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
            transition: left 0.5s ease;
        }
        
        .gradio-button:hover {
            background: linear-gradient(135deg, #5855eb 0%, #7c3aed 100%) !important;
            transform: translateY(-2px) !important;
            box-shadow: 
                0 8px 25px rgba(99, 102, 241, 0.35),
                0 2px 6px rgba(0, 0, 0, 0.15) !important;
        }
        
        .gradio-button:hover::before {
            left: 100%;
        }
        
        /* Primary Action Button */
        .gradio-button.primary {
            background: linear-gradient(135deg, #059669 0%, #10b981 50%, #34d399 100%) !important;
            box-shadow: 
                0 4px 15px rgba(16, 185, 129, 0.25),
                0 1px 3px rgba(0, 0, 0, 0.1) !important;
        }
        
        .gradio-button.primary:hover {
            background: linear-gradient(135deg, #047857 0%, #059669 50%, #10b981 100%) !important;
            box-shadow: 
                0 8px 25px rgba(16, 185, 129, 0.35),
                0 2px 6px rgba(0, 0, 0, 0.15) !important;
        }
        
        /* Secondary Buttons */
        .gradio-button.secondary {
            background: linear-gradient(135deg, #64748b 0%, #94a3b8 100%) !important;
            box-shadow: 
                0 4px 15px rgba(100, 116, 139, 0.2),
                0 1px 3px rgba(0, 0, 0, 0.08) !important;
        }
        
        .gradio-button.secondary:hover {
            background: linear-gradient(135deg, #475569 0%, #64748b 100%) !important;
            box-shadow: 
                0 8px 25px rgba(100, 116, 139, 0.3),
                0 2px 6px rgba(0, 0, 0, 0.12) !important;
        }
        
        /* Elegant DataFrame */
        .gradio-dataframe {
            background: linear-gradient(135deg, #ffffff 0%, #fefefe 100%) !important;
            border: 1px solid rgba(99, 102, 241, 0.1) !important;
            border-radius: 16px !important;
            overflow: hidden !important;
            box-shadow: 
                0 4px 20px rgba(99, 102, 241, 0.08),
                0 1px 3px rgba(0, 0, 0, 0.05) !important;
        }
        
        .gradio-dataframe table {
            background: transparent !important;
            font-family: 'Inter', sans-serif !important;
        }
        
        .gradio-dataframe th {
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%) !important;
            color: #1e293b !important;
            font-weight: 600 !important;
            font-family: 'Poppins', sans-serif !important;
            border-bottom: 2px solid #e2e8f0 !important;
            padding: 1rem 0.75rem !important;
        }
        
        .gradio-dataframe td {
            border-bottom: 1px solid #f1f5f9 !important;
            padding: 0.875rem 0.75rem !important;
            transition: background-color 0.2s ease !important;
        }
        
        .gradio-dataframe tr:hover td {
            background-color: rgba(99, 102, 241, 0.02) !important;
        }
        
        /* Status Messaging */
        .gradio-html {
            background: linear-gradient(135deg, #ffffff 0%, #fefefe 100%) !important;
            border: 1px solid rgba(99, 102, 241, 0.1) !important;
            border-radius: 12px !important;
            padding: 1.25rem !important;
            box-shadow: 
                0 2px 10px rgba(99, 102, 241, 0.05),
                0 1px 3px rgba(0, 0, 0, 0.03) !important;
            font-family: 'Inter', sans-serif !important;
        }
        
        /* File Upload Enhancement */
        .gradio-file {
            background: linear-gradient(135deg, #fafbfc 0%, #f8fafc 100%) !important;
            border: 2px dashed #cbd5e1 !important;
            border-radius: 12px !important;
            transition: all 0.3s ease !important;
        }
        
        .gradio-file:hover {
            border-color: #6366f1 !important;
            background: linear-gradient(135deg, #f0f4ff 0%, #e0e7ff 100%) !important;
            transform: translateY(-1px) !important;
        }
        
        /* Sophisticated Accordion */
        .gradio-accordion {
            background: linear-gradient(135deg, #ffffff 0%, #fefefe 100%) !important;
            border: 1px solid rgba(99, 102, 241, 0.08) !important;
            border-radius: 12px !important;
            overflow: hidden !important;
            box-shadow: 0 2px 8px rgba(99, 102, 241, 0.04) !important;
        }
        
        .gradio-accordion summary {
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%) !important;
            color: #1e293b !important;
            font-weight: 500 !important;
            font-family: 'Poppins', sans-serif !important;
            padding: 1.25rem !important;
            cursor: pointer !important;
            transition: all 0.2s ease !important;
        }
        
        .gradio-accordion summary:hover {
            background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%) !important;
        }
        
        /* Elegant Checkbox Groups */
        .gradio-checkboxgroup label {
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%) !important;
            border: 1.5px solid #e2e8f0 !important;
            border-radius: 10px !important;
            padding: 0.75rem 1rem !important;
            margin: 0.375rem !important;
            transition: all 0.3s ease !important;
            font-family: 'Inter', sans-serif !important;
            font-weight: 500 !important;
        }
        
        .gradio-checkboxgroup label:hover {
            background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%) !important;
            border-color: #6366f1 !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 2px 8px rgba(99, 102, 241, 0.15) !important;
        }
        
        .gradio-checkboxgroup input:checked + span {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
            color: white !important;
        }
        
        /* Console Log Styling */
        .gradio-textbox[readonly] {
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%) !important;
            border: 1px solid #475569 !important;
            color: #e2e8f0 !important;
            font-family: 'JetBrains Mono', 'Monaco', 'Menlo', 'Ubuntu Mono', monospace !important;
            font-size: 0.9rem !important;
            line-height: 1.6 !important;
        }
        
        /* Responsive Enhancements */
        @media (max-width: 768px) {
            .enterprise-title {
                font-size: 2.5rem !important;
            }
            
            .enterprise-subtitle {
                font-size: 1.1rem !important;
            }
            
            .gradio-group {
                padding: 1.5rem !important;
            }
            
            .gradio-button {
                padding: 0.875rem 1.5rem !important;
            }
        }
        
        /* Premium Loading States */
        @keyframes shimmer {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }
        
        .loading {
            position: relative;
            overflow: hidden;
        }
        
        .loading::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.1), transparent);
            animation: shimmer 2s infinite;
        }
        
        /* Status States */
        .status-success {
            background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%) !important;
            border: 1px solid #10b981 !important;
            color: #047857 !important;
        }
        
        .status-error {
            background: linear-gradient(135deg, #fef2f2 0%, #fecaca 100%) !important;
            border: 1px solid #ef4444 !important;
            color: #dc2626 !important;
        }
        
        .status-processing {
            background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%) !important;
            border: 1px solid #3b82f6 !important;
            color: #1d4ed8 !important;
        }
        
        /* Enhanced Typography */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Poppins', sans-serif !important;
        }
        
        p, span, div, label {
            font-family: 'Inter', sans-serif !important;
        }
        
        /* Elegant Shadows */
        .shadow-elegant {
            box-shadow: 
                0 4px 20px rgba(99, 102, 241, 0.08),
                0 1px 3px rgba(0, 0, 0, 0.05) !important;
        }
        
        .shadow-elegant-hover:hover {
            box-shadow: 
                0 8px 30px rgba(99, 102, 241, 0.12),
                0 2px 6px rgba(0, 0, 0, 0.08) !important;
        }
        """
    ) as app:
        
        # Sophisticated Header
        gr.HTML("""
            <div class="enterprise-header">
                <h1 class="enterprise-title">🎯 Enterprise Website Intelligence</h1>
                <p class="enterprise-subtitle">Advanced AI-powered business intelligence platform • Automated website analysis • Strategic insights & competitive intelligence</p>
            </div>
        """)
        
        # State variables
        csv_file = gr.State(None)
        all_results_state = gr.State(pd.DataFrame())  # FIX: Store accumulated results separately
        
        # Input Section
        with gr.Group():
            gr.HTML("<h3>⚙️ Intelligence Configuration</h3>")
            
            with gr.Row():
                with gr.Column(scale=3):
                    urls_input = gr.Textbox(
                        label="🌐 Target Websites",
                        placeholder="example.com\ncompany.org\nbusiness.net",
                        lines=4,
                        info="Enter one URL per line for comprehensive analysis"
                    )
                with gr.Column(scale=1):
                    file_upload = gr.File(
                        label="📄 Import URL List",
                        file_types=[".txt", ".csv"]
                    )
                    load_button = gr.Button("📥 Load File", size="sm")
            
            with gr.Row():
                business_terms_input = gr.Textbox(
                    label="🔍 Strategic Keywords",
                    placeholder="technology, manufacturing, consulting, finance",
                    info="Define business intelligence filters"
                )
                api_key_input = gr.Textbox(
                    label="🔐 AI API Key",
                    placeholder="sk-...",
                    type="password",
                    info="Required for advanced AI processing"
                )
            
            with gr.Row():
                customer_type_input = gr.CheckboxGroup(
                    choices=["Business to Business", "Business to Consumer", "Both"],
                    label="👥 Market Segments",
                    info="Filter by customer engagement model"
                )
                operation_type_input = gr.CheckboxGroup(
                    choices=["Manufacturing", "Trading", "Services"],
                    label="🏭 Business Models",
                    info="Filter by operational structure"
                )
            
            submit_button = gr.Button(
                "🚀 Execute Analysis",
                variant="primary",
                size="lg"
            )
        
        # Status and Results
        status_output = gr.HTML("🎯 Ready to deploy enterprise intelligence analysis")
        
        with gr.Group():
            gr.HTML("<h3>📊 Intelligence Dashboard</h3>")
            results_output = gr.DataFrame(
                interactive=False,
                wrap=False,
                height=600
            )
            
            with gr.Accordion("📋 Execution Log", open=False):
                log_output = gr.Textbox(
                    lines=10,
                    max_lines=20,
                    interactive=False
                )
        
        # Download Section
        with gr.Group():
            gr.HTML("<h3>📦 Intelligence Reports</h3>")
            
            with gr.Row():
                download_csv = gr.Button("📊 Export Data", variant="secondary")
                download_reports = gr.Button("📄 Generate Reports", variant="secondary")
                download_raw = gr.Button("🗂️ Raw Intelligence", variant="secondary")
            
            with gr.Row():
                csv_output = gr.File(label="Data Export", visible=False)
                reports_output = gr.File(label="Analysis Reports", visible=False)
                raw_output = gr.File(label="Raw Data", visible=False)
        
        # FIXED: Processing function with simpler approach
        def process_websites_ui(urls_text, business_terms_text, api_key, customer_types, operation_types):
            nonlocal processor
            
            if not urls_text.strip():
                yield "❌ No URLs provided", pd.DataFrame(), None, "Please enter target websites", pd.DataFrame()
                return
                
            if not api_key.strip():
                yield "🔑 API key required", pd.DataFrame(), None, "Please provide AI API key", pd.DataFrame()
                return
            
            urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
            business_terms = [term.strip() for term in business_terms_text.split(',') if term.strip()]
            
            # Initialize processor (this will clear previous results)
            processor = StreamingWebsiteProcessor(api_key, max_workers=4)
            
            for stream_result in processor.process_websites_stream(
                urls, business_terms, customer_types, operation_types
            ):
                # Handle both old and new yield formats
                if len(stream_result) == 5:
                    status, df_current, csv_file_path, log, is_final = stream_result
                else:
                    # Fallback for old format
                    status, df_current, csv_file_path, log = stream_result
                    is_final = False
                
                # Get the current complete state from processor
                current_complete_df = processor._results_to_dataframe()
                
                # Always yield the complete current state
                yield status, current_complete_df, csv_file_path, log, current_complete_df
        
        def load_urls_from_file(file):
            if file is None:
                return ""
            try:
                content = file.decode('utf-8')
                return content
            except Exception as e:
                return f"Error reading file: {str(e)}"
        
        # FIX: Proper download handlers
        def handle_csv_download(all_results_df):
            if all_results_df is not None and not all_results_df.empty:
                try:
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    temp_dir = tempfile.mkdtemp()
                    csv_filename = os.path.join(temp_dir, f"enterprise_analysis_{timestamp}.csv")
                    all_results_df.to_csv(csv_filename, index=False, escapechar='\\', quoting=1)
                    return gr.update(value=csv_filename, visible=True)
                except Exception as e:
                    logger.error(f"Error creating CSV: {str(e)}")
                    return gr.update(visible=False)
            return gr.update(visible=False)
        
        def handle_reports_download():
            nonlocal processor
            if processor and processor.results:
                try:
                    reports_file = processor._save_detailed_reports()
                    if reports_file:
                        return gr.update(value=reports_file, visible=True)
                except Exception as e:
                    logger.error(f"Error generating reports: {str(e)}")
            return gr.update(visible=False)
        
        def handle_raw_data_download():
            nonlocal processor
            if processor and processor.results:
                try:
                    raw_file = processor._save_raw_data()
                    if raw_file:
                        return gr.update(value=raw_file, visible=True)
                except Exception as e:
                    logger.error(f"Error generating raw data: {str(e)}")
            return gr.update(visible=False)
        
        # Event handlers
        submit_button.click(
            fn=process_websites_ui,
            inputs=[urls_input, business_terms_input, api_key_input, customer_type_input, operation_type_input],
            outputs=[status_output, results_output, csv_file, log_output, all_results_state]
        )
        
        load_button.click(
            fn=load_urls_from_file,
            inputs=[file_upload],
            outputs=[urls_input]
        )
        
        # FIX: Proper download event handlers
        download_csv.click(
            fn=handle_csv_download,
            inputs=[all_results_state],
            outputs=[csv_output]
        )
        
        download_reports.click(
            fn=handle_reports_download,
            inputs=[],
            outputs=[reports_output]
        )
        
        download_raw.click(
            fn=handle_raw_data_download,
            inputs=[],
            outputs=[raw_output]
        )
    
    return app

# Main execution
if __name__ == "__main__":
    print("🎯 Starting Enterprise Website Intelligence Platform")
    print("=" * 60)
    
    # Check dependencies
    missing_deps = []
    
    try:
        import langchain
        print("✅ Core AI framework available")
    except ImportError:
        missing_deps.append("langchain")
    
    try:
        import openai
        print("✅ AI processing engine available")
    except ImportError:
        missing_deps.append("openai")
    
    if LANGGRAPH_AVAILABLE:
        print("✅ Advanced workflow engine enabled")
    else:
        print("⚠️  Advanced workflows not available - Using standard processing")
        print("   Enhanced features: pip install langgraph")
    
    if missing_deps:
        print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        print("Install with: pip install langchain langchain-openai")
        exit(1)
    
    print("🌐 Initializing enterprise interface...")
    print("=" * 60)
    
    # Test basic functionality
    try:
        test_analyzer = AdvancedWebsiteAnalyzer("test-key")
        print("✅ Intelligence engine initialization successful")
    except Exception as e:
        print(f"⚠️  Intelligence engine initialization warning: {e}")
    
    app = create_enterprise_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7863,
        share=False,
        inbrowser=True,
        show_error=True
    )
