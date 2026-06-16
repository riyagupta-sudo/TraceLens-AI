import os
import sys

# Adjust path to import from app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

def test_provider_configuration():
    print("=" * 60)
    print("RUNNING OSINT PROVIDER CONFIGURATION VERIFICATION")
    print("=" * 60)

    # 1. Load .env
    app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dotenv_path = os.path.join(app_dir, ".env")
    print(f"Loading .env from: {dotenv_path}")
    load_dotenv(dotenv_path)

    # 2. Check all provider keys via get_provider_availability()
    from app.web_intelligence import get_provider_availability
    availability = get_provider_availability()

    print("\n[Availability Status]")
    for key, value in availability.items():
        print(f"  {key:<15}: {'AVAILABLE' if value else 'MISSING'}")

    # Verify return schema is correct
    assert isinstance(availability, dict)
    for k in ["apify", "google_lens", "bing_visual", "yandex", "tineye"]:
        assert k in availability
        assert isinstance(availability[k], bool)

    # 3. Verify provider initialization succeeds and missing keys do not throw exceptions
    print("\n[Initializing Web Intelligence Providers]")
    from app.web_intelligence import (
        GoogleLensProvider as WebGoogleLens,
        BingVisualProvider as WebBingVisual,
        YandexProvider as WebYandex,
        TinEyeProvider as WebTinEye,
        MockProvider as WebMock,
        ProviderRegistry
    )

    try:
        g_lens = WebGoogleLens()
        print("  Web GoogleLensProvider ..... INITIALIZED")
        bing = WebBingVisual()
        print("  Web BingVisualProvider ..... INITIALIZED")
        yandex = WebYandex()
        print("  Web YandexProvider ......... INITIALIZED")
        tineye = WebTinEye()
        print("  Web TinEyeProvider ......... INITIALIZED")
        mock = WebMock()
        print("  Web MockProvider ........... INITIALIZED")
        registry = ProviderRegistry()
        print("  ProviderRegistry ........... INITIALIZED")
    except Exception as e:
        assert False, f"Web intelligence provider initialization failed with exception: {e}"

    print("\n[Initializing Manual Hunt OSINT Providers]")
    from app.osint_intelligence import (
        GoogleLensProvider as HuntGoogleLens,
        BingVisualSearchProvider as HuntBingVisual,
        YandexProvider as HuntYandex,
        TinEyeProvider as HuntTinEye,
        OSINTProviderManager
    )

    try:
        hunt_g_lens = HuntGoogleLens()
        print("  Hunt GoogleLensProvider .... INITIALIZED")
        hunt_bing = HuntBingVisual()
        print("  Hunt BingVisualSearch ...... INITIALIZED")
        hunt_yandex = HuntYandex()
        print("  Hunt YandexProvider ........ INITIALIZED")
        hunt_tineye = HuntTinEye()
        print("  Hunt TinEyeProvider ........ INITIALIZED")
        manager = OSINTProviderManager()
        print("  OSINTProviderManager ....... INITIALIZED")
    except Exception as e:
        assert False, f"OSINT provider initialization failed with exception: {e}"

    # Verify that missing credentials do not throw exceptions when initialized
    print("\nVerifying initialization under simulated missing credentials...")
    original_env = {}
    keys_to_clear = [
        "APIFY_TOKEN", "APIFY_API_TOKEN",
        "GOOGLE_LENS_API_KEY",
        "BING_VISUAL_SEARCH_API_KEY",
        "SERPAPI_API_KEY",
        "TINEYE_API_KEY"
    ]
    for key in keys_to_clear:
        if key in os.environ:
            original_env[key] = os.environ[key]
            del os.environ[key]

    try:
        # Re-verify availability registers all as false
        cleared_availability = get_provider_availability()
        for k in cleared_availability:
            assert not cleared_availability[k], f"Expected {k} to be False when environment variables are cleared"

        # Re-initialize
        WebGoogleLens()
        WebBingVisual()
        WebYandex()
        WebTinEye()
        WebMock()
        ProviderRegistry()
        HuntGoogleLens()
        HuntBingVisual()
        HuntYandex()
        HuntTinEye()
        OSINTProviderManager()
        print("  All providers successfully initialized with empty environment variables.")
    except Exception as e:
        assert False, f"Provider initialization threw an error when credentials were missing: {e}"
    finally:
        # Restore environment
        for key, value in original_env.items():
            os.environ[key] = value

    print("\n" + "=" * 60)
    print("ALL OSINT PROVIDER CONFIGURATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    test_provider_configuration()
