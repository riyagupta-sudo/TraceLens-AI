import os
from dotenv import load_dotenv

# Load environment variables relative to this file
app_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(app_dir)
load_dotenv(os.path.join(backend_dir, ".env"))

import time
import random
import asyncio
import datetime
from typing import Dict, Any, List, Optional
import httpx
import numpy as np
from sqlalchemy.orm import Session
from .models import OSINTScan, OSINTResult, MediaItem

def get_provider_availability() -> dict:
    """Returns a dictionary mapping each provider name to its configured status."""
    return {
        "apify": bool(os.getenv("APIFY_TOKEN") or os.getenv("APIFY_API_TOKEN")),
        "google_lens": bool(os.getenv("GOOGLE_LENS_API_KEY")),
        "bing_visual": bool(os.getenv("BING_VISUAL_SEARCH_API_KEY")),
        "yandex": bool(os.getenv("SERPAPI_API_KEY")),
        "tineye": bool(os.getenv("TINEYE_API_KEY"))
    }


class WebIntelligenceProvider:
    """Base interface for all web reverse image search providers."""
    async def query(self, filepath: str, phash: str, embedding: List[float], timeout_sec: float = 5.0) -> Dict[str, Any]:
        raise NotImplementedError("Providers must implement this method")

class MockProvider(WebIntelligenceProvider):
    """Simulates realistic web intelligence using registered database metadata and query signatures."""
    async def query(self, filepath: str, phash: str, embedding: List[float], timeout_sec: float = 5.0) -> Dict[str, Any]:
        # Simulate network latency
        await asyncio.sleep(0.1)
        
        filename = os.path.basename(filepath).lower()
        
        # Registry mapping filenames to mock responses
        registry = {
            "pahalgam": {
                "earliest_known_appearance": "2021-05-12T09:30:00Z",
                "known_domains": ["travelphotography.org", "wikipedia.org", "flickr.com", "kashmirtourism.gov.in"],
                "possible_original_source": "https://travelphotography.org/gallery/pahalgam_valley_2021.jpg",
                "confidence": 92,
                "discovered_variants": [
                    {
                        "url": "https://travelphotography.org/gallery/pahalgam_valley_2021.jpg",
                        "domain": "travelphotography.org",
                        "first_seen": "2021-05-12T09:30:00Z",
                        "title": "Beautiful Pahalgam Valley Original",
                        "similarity": 98,
                        "snippet": "High resolution landscape photograph of Pahalgam valley taken with DSLR.",
                        "source_type": "heuristic"
                    },
                    {
                        "url": "https://en.wikipedia.org/wiki/Pahalgam#/media/File:Pahalgam_Valley.png",
                        "domain": "wikipedia.org",
                        "first_seen": "2022-03-24T14:15:00Z",
                        "title": "Pahalgam - Wikipedia entry",
                        "similarity": 92,
                        "snippet": "Medium resolution cropped version used as the main Wikipedia image for Pahalgam page.",
                        "source_type": "heuristic"
                    }
                ]
            },
            "drone": {
                "earliest_known_appearance": "2023-08-20T14:22:00Z",
                "known_domains": ["defence-blog.com", "leakleak.net", "intel-leak.org"],
                "possible_original_source": "https://intel-leak.org/files/drone_orignal.jpg",
                "confidence": 96,
                "discovered_variants": [
                    {
                        "url": "https://intel-leak.org/files/drone_orignal.jpg",
                        "domain": "intel-leak.org",
                        "first_seen": "2023-08-20T14:22:00Z",
                        "title": "Confidential Drone Imagery Raw",
                        "similarity": 100,
                        "snippet": "Original 12MP high fidelity telemetry snapshot leaked on forums.",
                        "source_type": "heuristic"
                    }
                ]
            },
            "human": {
                "earliest_known_appearance": "2022-11-05T09:12:00Z",
                "known_domains": ["faceprofile.org", "socialnet.com"],
                "possible_original_source": "https://socialnet.com/users/profile_pic.jpg",
                "confidence": 88,
                "discovered_variants": [
                    {
                        "url": "https://socialnet.com/users/profile_pic.jpg",
                        "domain": "socialnet.com",
                        "first_seen": "2022-11-05T09:12:00Z",
                        "title": "SocialNet User Profile Picture",
                        "similarity": 98,
                        "snippet": "Original portrait upload, size 600x900.",
                        "source_type": "heuristic"
                    }
                ]
            },
            "building": {
                "earliest_known_appearance": "2023-04-10T10:15:00Z",
                "known_domains": ["buildingdb.com", "architecture-forum.org"],
                "possible_original_source": "https://buildingdb.com/media/building_001.jpg",
                "confidence": 85,
                "discovered_variants": [
                    {
                        "url": "https://buildingdb.com/media/building_001.jpg",
                        "domain": "buildingdb.com",
                        "first_seen": "2023-04-10T10:15:00Z",
                        "title": "Urban Architecture Catalog Building 001",
                        "similarity": 98,
                        "snippet": "Original database entry with architectural details.",
                        "source_type": "heuristic"
                    }
                ]
            }
        }
        
        # Check matching key
        for key in registry:
            if key in filename:
                return registry[key]
                
        # Dynamic fallback for general images using phash as seed
        random.seed(phash or filename)
        days_ago = random.randint(100, 1000)
        first_date = (datetime.datetime.utcnow() - datetime.timedelta(days=days_ago)).isoformat() + "Z"
        
        domain = random.choice(["deviantart.com", "unsplash.com", "pxhere.com", "shutterstock.com"])
        possible_source = f"https://{domain}/media/source_{random.randint(10000, 99999)}.jpg"
        
        return {
            "earliest_known_appearance": first_date,
            "known_domains": [domain],
            "possible_original_source": possible_source,
            "confidence": random.randint(60, 90),
            "discovered_variants": [
                {
                    "url": possible_source,
                    "domain": domain,
                    "first_seen": first_date,
                    "title": "Stock Visual Match entry",
                    "similarity": random.randint(75, 95),
                    "snippet": "Perceptually similar image index match.",
                    "source_type": "simulated"
                }
            ]
        }

class GoogleLensProvider(WebIntelligenceProvider):
    """Google Lens Visual Search Provider."""
    async def query(self, filepath: str, phash: str, embedding: List[float], timeout_sec: float = 5.0) -> Dict[str, Any]:
        api_key = os.getenv("GOOGLE_LENS_API_KEY")
        if not api_key:
            return {"provider": "Google Lens", "status": "Disabled", "reason": "API key not configured"}
        
        uploads_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
        actual_path = os.path.join(uploads_dir, os.path.basename(filepath))
        if not os.path.exists(actual_path):
            actual_path = filepath
            
        try:
            async with httpx.AsyncClient() as client:
                with open(actual_path, "rb") as f:
                    resp = await client.post(
                        "https://serpapi.com/search.json",
                        files={"image": f},
                        data={"engine": "google_lens", "api_key": api_key},
                        timeout=timeout_sec
                    )
                if resp.status_code == 200:
                    data = resp.json()
                    variants = []
                    domains = []
                    for item in data.get("visual_matches", []):
                        link = item.get("link")
                        if link:
                            domain = item.get("source", "Google Lens")
                            domains.append(domain)
                            variants.append({
                                "url": link,
                                "domain": domain,
                                "first_seen": None,
                                "title": item.get("title") or "Visual Match",
                                "similarity": int(item.get("similarity", 80)),
                                "snippet": item.get("title") or "",
                                "source_type": "real_provider"
                            })
                    return {
                        "earliest_known_appearance": None,
                        "known_domains": list(set(domains)),
                        "possible_original_source": variants[0]["url"] if variants else None,
                        "confidence": int(sum(v["similarity"] for v in variants)/len(variants)) if variants else 0,
                        "discovered_variants": variants
                    }
        except Exception as e:
            print(f"[GOOGLE LENS ERROR] {e}")
        return {"earliest_known_appearance": None, "known_domains": [], "possible_original_source": None, "confidence": 0, "discovered_variants": []}

class TinEyeProvider(WebIntelligenceProvider):
    """TinEye Visual Search Provider."""
    async def query(self, filepath: str, phash: str, embedding: List[float], timeout_sec: float = 5.0) -> Dict[str, Any]:
        api_key = os.getenv("TINEYE_API_KEY")
        if not api_key:
            return {"provider": "TinEye", "status": "Disabled", "reason": "API key not configured"}
        
        uploads_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
        actual_path = os.path.join(uploads_dir, os.path.basename(filepath))
        if not os.path.exists(actual_path):
            actual_path = filepath
            
        try:
            async with httpx.AsyncClient() as client:
                with open(actual_path, "rb") as f:
                    resp = await client.post(
                        "https://api.tineye.com/rest/search/",
                        params={"api_key": api_key},
                        files={"image": f},
                        timeout=timeout_sec
                    )
                if resp.status_code == 200:
                    data = resp.json()
                    variants = []
                    domains = []
                    for item in data.get("results", []):
                        backlinks = item.get("backlinks", [])
                        first_seen = backlinks[0].get("crawl_date") if backlinks else None
                        url = backlinks[0].get("url") if backlinks else item.get("image_url")
                        if url:
                            domain = item.get("domain", "TinEye")
                            domains.append(domain)
                            variants.append({
                                "url": url,
                                "domain": domain,
                                "first_seen": first_seen,
                                "title": item.get("top_level_domain") or "TinEye Match",
                                "similarity": int(item.get("score", 80)),
                                "snippet": f"TinEye match from domain: {domain}",
                                "source_type": "real_provider"
                              })
                    
                    dates = [v["first_seen"] for v in variants if v.get("first_seen")]
                    earliest = min(dates) if dates else None
                    
                    return {
                        "earliest_known_appearance": earliest,
                        "known_domains": list(set(domains)),
                        "possible_original_source": variants[0]["url"] if variants else None,
                        "confidence": int(sum(v["similarity"] for v in variants)/len(variants)) if variants else 0,
                        "discovered_variants": variants
                    }
        except Exception as e:
            print(f"[TINEYE ERROR] {e}")
        return {"earliest_known_appearance": None, "known_domains": [], "possible_original_source": None, "confidence": 0, "discovered_variants": []}

class YandexProvider(WebIntelligenceProvider):
    """Yandex Images Visual Search Provider."""
    async def query(self, filepath: str, phash: str, embedding: List[float], timeout_sec: float = 5.0) -> Dict[str, Any]:
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            return {"provider": "Yandex Images", "status": "Disabled", "reason": "API key not configured"}
        
        uploads_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
        actual_path = os.path.join(uploads_dir, os.path.basename(filepath))
        if not os.path.exists(actual_path):
            actual_path = filepath
            
        try:
            async with httpx.AsyncClient() as client:
                with open(actual_path, "rb") as f:
                    resp = await client.post(
                        "https://serpapi.com/search.json",
                        files={"image": f},
                        data={"engine": "yandex_images", "api_key": api_key},
                        timeout=timeout_sec
                    )
                if resp.status_code == 200:
                    data = resp.json()
                    variants = []
                    domains = []
                    for item in data.get("similar_images", []):
                        link = item.get("link")
                        if link:
                            domain = item.get("source", "Yandex")
                            domains.append(domain)
                            variants.append({
                                "url": link,
                                "domain": domain,
                                "first_seen": None,
                                "title": item.get("title") or "Yandex Match",
                                "similarity": 80,
                                "snippet": item.get("title") or "",
                                "source_type": "real_provider"
                            })
                    return {
                        "earliest_known_appearance": None,
                        "known_domains": list(set(domains)),
                        "possible_original_source": variants[0]["url"] if variants else None,
                        "confidence": 80 if variants else 0,
                        "discovered_variants": variants
                    }
        except Exception as e:
            print(f"[YANDEX ERROR] {e}")
        return {"earliest_known_appearance": None, "known_domains": [], "possible_original_source": None, "confidence": 0, "discovered_variants": []}

class BingVisualProvider(WebIntelligenceProvider):
    """Bing Visual Search Provider."""
    async def query(self, filepath: str, phash: str, embedding: List[float], timeout_sec: float = 5.0) -> Dict[str, Any]:
        api_key = os.getenv("BING_VISUAL_SEARCH_API_KEY")
        if not api_key:
            return {"provider": "Bing Visual Search", "status": "Disabled", "reason": "API key not configured"}
        
        uploads_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
        actual_path = os.path.join(uploads_dir, os.path.basename(filepath))
        if not os.path.exists(actual_path):
            actual_path = filepath
            
        try:
            async with httpx.AsyncClient() as client:
                headers = {"Ocp-Apim-Subscription-Key": api_key}
                with open(actual_path, "rb") as f:
                    resp = await client.post(
                        "https://api.bing.microsoft.com/v7.0/images/visualsearch",
                        headers=headers,
                        files={"image": f},
                        timeout=timeout_sec
                    )
                if resp.status_code == 200:
                    data = resp.json()
                    variants = []
                    domains = []
                    for tag in data.get("tags", []):
                        for action in tag.get("actions", []):
                            if action.get("actionType") == "VisualSearch":
                                for item in action.get("data", {}).get("value", []):
                                    url = item.get("hostPageUrl")
                                    if url:
                                        domain = item.get("hostPageDomainFriendly") or "Bing"
                                        domains.append(domain)
                                        variants.append({
                                            "url": url,
                                            "domain": domain,
                                            "first_seen": item.get("datePublished"),
                                            "title": item.get("name") or "Bing Match",
                                            "similarity": 85,
                                            "snippet": item.get("name") or "",
                                            "source_type": "real_provider"
                                        })
                    
                    dates = [v["first_seen"] for v in variants if v.get("first_seen")]
                    earliest = min(dates) if dates else None
                    
                    return {
                        "earliest_known_appearance": earliest,
                        "known_domains": list(set(domains)),
                        "possible_original_source": variants[0]["url"] if variants else None,
                        "confidence": 85 if variants else 0,
                        "discovered_variants": variants
                    }
        except Exception as e:
            print(f"[BING VISUAL ERROR] {e}")
        return {"earliest_known_appearance": None, "known_domains": [], "possible_original_source": None, "confidence": 0, "discovered_variants": []}

class ProviderRegistry:
    """Manages active intelligence providers and isolates failures."""
    def __init__(self):
        availability = get_provider_availability()
        any_real_keys = any([
            availability["google_lens"],
            availability["bing_visual"],
            availability["yandex"],
            availability["tineye"]
        ])
        
        self.providers = []
        if not any_real_keys:
            self.providers.append(("Mock", MockProvider(), 1.0))
            
        self.providers.extend([
            ("GoogleLens", GoogleLensProvider(), 1.0),
            ("TinEye", TinEyeProvider(), 1.0),
            ("Yandex", YandexProvider(), 1.0),
            ("BingVisual", BingVisualProvider(), 1.0)
        ])
        
    async def aggregate_queries(self, filepath: str, phash: str, embedding: List[float]) -> Dict[str, Any]:
        combined_results = {
            "earliest_known_appearance": None,
            "known_domains": [],
            "discovered_variants": [],
            "possible_original_source": None,
            "confidence": 0
        }
        
        active_dates = []
        active_confidences = []
        
        for name, provider, weight in self.providers:
            try:
                res = await asyncio.wait_for(
                    provider.query(filepath, phash, embedding),
                    timeout=5.0
                )
                
                if isinstance(res, dict) and res.get("status") == "Disabled":
                    continue
                
                if res.get("earliest_known_appearance"):
                    active_dates.append(res["earliest_known_appearance"])
                if res.get("confidence") and res["confidence"] > 0:
                    active_confidences.append(res["confidence"] * weight)
                if res.get("known_domains"):
                    combined_results["known_domains"].extend(res["known_domains"])
                if res.get("discovered_variants"):
                    combined_results["discovered_variants"].extend(res["discovered_variants"])
                if res.get("possible_original_source") and not combined_results["possible_original_source"]:
                    combined_results["possible_original_source"] = res["possible_original_source"]
                    
            except Exception as e:
                print(f"[OSINT PROVIDER WARNING] Provider {name} failed: {e}")
                
        if active_dates:
            parsed_dates = []
            for d in active_dates:
                try:
                    parsed_dates.append(datetime.datetime.fromisoformat(d.replace("Z", "")))
                except Exception:
                    pass
            if parsed_dates:
                earliest_dt = min(parsed_dates)
                combined_results["earliest_known_appearance"] = earliest_dt.isoformat() + "Z"
                
        if active_confidences:
            combined_results["confidence"] = int(np.mean(active_confidences)) if 'np' in globals() else int(sum(active_confidences)/len(active_confidences))
        else:
            combined_results["confidence"] = 50
            
        combined_results["known_domains"] = list(set(combined_results["known_domains"]))
        
        return combined_results

# Background runner
async def run_asynchronous_web_intelligence(media_id: int, db_session_factory):
    """Queries reverse-image intelligence asynchronously and writes results to DB."""
    db = db_session_factory()
    try:
        # Load item
        item = db.query(MediaItem).filter(MediaItem.id == media_id).first()
        if not item:
            print(f"[ASYNC OSINT ERROR] MediaItem with ID {media_id} not found.")
            return
            
        print(f"[ASYNC OSINT START] Processing reverse image search for: {item.filename}")
        
        # Build registry and query providers
        registry = ProviderRegistry()
        web_intel = await registry.aggregate_queries(item.filepath, item.phash, item.embedding)
        
        # Update OSINT scan status to Completed
        scan = db.query(OSINTScan).filter(OSINTScan.media_id == media_id).first()
        if not scan:
            scan = OSINTScan(media_id=media_id)
            db.add(scan)
        
        # Determine evidence-driven state
        availability = get_provider_availability()
        any_real_keys = any([
            availability["google_lens"],
            availability["bing_visual"],
            availability["yandex"],
            availability["tineye"]
        ])
        
        has_real_evidence = any(v.get("source_type") == "real_provider" for v in web_intel.get("discovered_variants", []))
        
        if not any_real_keys:
            scan.status = "Provider Unavailable"
            status_msg = "Provider Unavailable"
        elif has_real_evidence:
            scan.status = "Verified Matches Found"
            status_msg = "Verified Matches Found"
        else:
            scan.status = "No Matches Found"
            status_msg = "No Matches Found"
            
        scan.tags = web_intel["known_domains"]
        db.commit()
        
        # Add discovered variants to OSINTResult table
        # Clear existing ones first
        db.query(OSINTResult).filter(OSINTResult.media_id == media_id).delete()
        
        for variant in web_intel.get("discovered_variants", []):
            result = OSINTResult(
                media_id=media_id,
                url=variant["url"],
                title=variant.get("title", "Discovered Variant"),
                snippet=variant.get("snippet", "Visual duplicate matched online"),
                publication_date=variant.get("first_seen"),
                source=variant.get("domain", "Web"),
                confidence=variant.get("similarity", 80),
                reason="Discovered via reverse image search comparison",
                source_type=variant.get("source_type", "simulated")
            )
            db.add(result)
            
        # Update metadata signature with web intelligence properties
        meta = dict(item.metadata_sig or {})
        meta["earliest_known_appearance"] = web_intel["earliest_known_appearance"]
        meta["possible_original_source"] = web_intel["possible_original_source"]
        meta["web_confidence"] = web_intel["confidence"]
        item.metadata_sig = meta
        
        # Update modification_report with osint_summary
        report = dict(item.modification_report or {})
        
        provider_status = {
            "Google Lens": "Active" if availability["google_lens"] else "Degraded / Token Missing",
            "Bing Visual Search": "Active" if availability["bing_visual"] else "Degraded / Token Missing",
            "Yandex": "Active" if availability["yandex"] else "Degraded / Token Missing",
            "TinEye": "Active" if availability["tineye"] else "Degraded / Token Missing"
        }
        if not any_real_keys:
            provider_status["Mock Registry"] = "Active"
            
        report["osint_summary"] = {
            "confidence_score": web_intel["confidence"],
            "match_count": len(web_intel.get("discovered_variants", [])),
            "first_seen": web_intel["earliest_known_appearance"] or "Unknown",
            "last_seen": "Unknown",
            "provider_status": provider_status,
            "known_domains": web_intel.get("known_domains", []),
            "status_message": status_msg
        }
        
        item.modification_report = report
        from sqlalchemy.orm.attributes import flag_modified
        flag_modified(item, "modification_report")
        
        # Recalculate forensic scores now that web intelligence chronology might have changed
        # This will run automatically in the database commit
        db.commit()
        print(f"[ASYNC OSINT SUCCESS] Processing completed for {item.filename}. Findings saved.")
        
    except Exception as e:
        print(f"[ASYNC OSINT ERROR] Processing failed for ID {media_id}: {e}")
        scan = db.query(OSINTScan).filter(OSINTScan.media_id == media_id).first()
        if scan:
            scan.status = "Failed"
            scan.error_message = str(e)
            db.commit()
    finally:
        db.close()
