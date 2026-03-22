"""
Tests for CHHAT Cigarette Brand Analyzer core modules.
"""

import json
import time
from unittest.mock import patch, MagicMock

import pytest

# ═══════════════════════════════════════════════════════════════════════════════
# brands.py tests
# ═══════════════════════════════════════════════════════════════════════════════

from brands import format_q12a, get_brand_list_for_prompt, BRANDS_AND_SKUS, BRAND_KHMER


class TestBrands:
    def test_format_q12a_produces_correct_format_with_khmer(self):
        result = format_q12a(["MEVIUS", "ARA"])
        assert "MEVIUS_ម៉ៃសេវែន / មេវៀស" in result
        assert "ARA_សេក" in result
        assert " | " in result

    def test_format_q12a_single_brand(self):
        result = format_q12a(["555"])
        assert "555_បារី​​ 555" in result
        assert " | " not in result

    def test_format_q12a_empty_list(self):
        result = format_q12a([])
        assert result == ""

    def test_format_q12a_unknown_brand_no_khmer(self):
        result = format_q12a(["UNKNOWN_BRAND"])
        assert result == "UNKNOWN_BRAND"

    def test_get_brand_list_for_prompt_includes_all_29_brands(self):
        prompt = get_brand_list_for_prompt()
        for brand in BRANDS_AND_SKUS:
            assert f"- {brand}:" in prompt
        # Verify all 29 brands are present
        assert len(BRANDS_AND_SKUS) == 29

    def test_get_brand_list_for_prompt_includes_skus(self):
        prompt = get_brand_list_for_prompt()
        assert "MEVIUS ORIGINAL" in prompt
        assert "ARA RED" in prompt

    def test_all_brands_have_khmer_entries(self):
        for brand in BRANDS_AND_SKUS:
            assert brand in BRAND_KHMER, f"Brand '{brand}' missing from BRAND_KHMER"

    def test_brand_khmer_values_are_nonempty(self):
        for brand, khmer in BRAND_KHMER.items():
            assert khmer, f"BRAND_KHMER['{brand}'] is empty"


# ═══════════════════════════════════════════════════════════════════════════════
# confidence.py tests
# ═══════════════════════════════════════════════════════════════════════════════

from confidence import compute_confidence


class TestConfidence:
    @patch("confidence.load_corrections", return_value=[])
    def test_high_confidence_for_perfect_conditions(self, mock_corr):
        result = compute_confidence(
            ai_confidence="high",
            brands_found=["MEVIUS", "ARA"],
            skus_found=["MEVIUS ORIGINAL", "ARA RED"],
            unidentified_packs=0,
            num_images=1,
            brands_per_image=None,
        )
        assert result["level"] == "high"
        assert result["score"] >= 80

    @patch("confidence.load_corrections", return_value=[])
    def test_confidence_drops_with_unidentified_packs(self, mock_corr):
        perfect = compute_confidence(
            ai_confidence="high",
            brands_found=["MEVIUS"],
            skus_found=["MEVIUS ORIGINAL"],
            unidentified_packs=0,
            num_images=1,
        )
        imperfect = compute_confidence(
            ai_confidence="high",
            brands_found=["MEVIUS"],
            skus_found=["MEVIUS ORIGINAL"],
            unidentified_packs=3,
            num_images=1,
        )
        assert imperfect["score"] < perfect["score"]

    @patch("confidence.load_corrections", return_value=[])
    def test_confidence_drops_with_poor_multi_image_consistency(self, mock_corr):
        # Good consistency: both images see the same brands
        consistent = compute_confidence(
            ai_confidence="high",
            brands_found=["MEVIUS", "ARA"],
            skus_found=["MEVIUS ORIGINAL", "ARA RED"],
            unidentified_packs=0,
            num_images=2,
            brands_per_image=[["MEVIUS", "ARA"], ["MEVIUS", "ARA"]],
        )
        # Poor consistency: images see completely different brands
        inconsistent = compute_confidence(
            ai_confidence="high",
            brands_found=["MEVIUS", "ARA"],
            skus_found=["MEVIUS ORIGINAL", "ARA RED"],
            unidentified_packs=0,
            num_images=2,
            brands_per_image=[["MEVIUS"], ["ARA"]],
        )
        assert inconsistent["score"] < consistent["score"]

    @patch("confidence.load_corrections", return_value=[])
    def test_score_between_0_and_100(self, mock_corr):
        # Test with worst possible conditions
        result = compute_confidence(
            ai_confidence="low",
            brands_found=["MEVIUS"],
            skus_found=[],
            unidentified_packs=10,
            num_images=3,
            brands_per_image=[["MEVIUS"], ["ARA"], ["555"]],
        )
        assert 0 <= result["score"] <= 100

    @patch("confidence.load_corrections", return_value=[])
    def test_result_has_required_keys(self, mock_corr):
        result = compute_confidence(
            ai_confidence="medium",
            brands_found=["MEVIUS"],
            skus_found=[],
            unidentified_packs=0,
            num_images=1,
        )
        assert "score" in result
        assert "level" in result
        assert "factors" in result
        assert result["level"] in ("high", "medium", "low")


# ═══════════════════════════════════════════════════════════════════════════════
# corrections.py tests
# ═══════════════════════════════════════════════════════════════════════════════

from corrections import (
    save_correction,
    load_corrections,
    find_relevant_corrections,
    format_corrections_for_prompt,
)


class TestCorrections:
    @pytest.fixture(autouse=True)
    def _use_temp_db(self, tmp_path, monkeypatch):
        """Use a temporary SQLite database for each test."""
        import database
        db_path = tmp_path / "test.db"
        monkeypatch.setattr(database, "DB_PATH", db_path)
        # Reset the cached connection so it reconnects to the temp DB
        monkeypatch.setattr(database, "_connection", None)

    def test_save_and_load_roundtrip(self):
        correction = {
            "serial": "001",
            "image_url": "https://example.com/img.jpg",
            "model_used": "gemini-2.5-pro",
            "ai_result": {"brands": ["MEVIUS"], "skus": ["MEVIUS ORIGINAL"]},
            "corrected_result": {"brands": ["MEVIUS", "ARA"], "skus": ["MEVIUS ORIGINAL", "ARA RED"]},
            "notes": "missed ARA pack",
        }
        save_correction(correction)
        loaded = load_corrections()
        assert len(loaded) == 1
        assert loaded[0]["serial"] == "001"
        assert loaded[0]["corrected_result"]["brands"] == ["MEVIUS", "ARA"]
        assert "id" in loaded[0]
        assert "timestamp" in loaded[0]

    def test_load_corrections_empty_file(self):
        result = load_corrections()
        assert result == []

    def test_find_relevant_corrections_returns_results(self):
        for brand in ["MEVIUS", "ARA", "555"]:
            save_correction({
                "serial": "001",
                "ai_result": {"brands": [brand], "skus": []},
                "corrected_result": {"brands": [brand, "ESSE"], "skus": []},
                "notes": "",
            })

        results = find_relevant_corrections(ai_brands=["MEVIUS"], limit=5)
        assert len(results) > 0
        found_brands = set()
        for r in results:
            found_brands.update(r.get("ai_result", {}).get("brands", []))
        assert "MEVIUS" in found_brands

    def test_find_relevant_corrections_without_ai_brands(self):
        save_correction({
            "serial": "001",
            "ai_result": {"brands": ["MEVIUS"], "skus": []},
            "corrected_result": {"brands": ["ARA"], "skus": []},
            "notes": "",
        })
        results = find_relevant_corrections(ai_brands=None, limit=5)
        assert len(results) == 1

    def test_format_corrections_for_prompt_produces_valid_text(self):
        save_correction({
            "serial": "001",
            "ai_result": {"brands": ["MEVIUS"], "skus": ["MEVIUS ORIGINAL"]},
            "corrected_result": {"brands": ["ARA"], "skus": ["ARA RED"]},
            "notes": "AI confused brands",
        })
        corrections = load_corrections()
        text = format_corrections_for_prompt(corrections)
        assert "LEARNING FROM PAST CORRECTIONS" in text
        assert "MEVIUS" in text
        assert "ARA" in text
        assert "AI confused brands" in text

    def test_format_corrections_for_prompt_empty(self):
        text = format_corrections_for_prompt([])
        assert text == ""

    def test_correction_type_brand_swap(self):
        save_correction({
            "serial": "001",
            "ai_result": {"brands": ["MEVIUS"], "skus": []},
            "corrected_result": {"brands": ["ARA"], "skus": []},
            "notes": "",
        })
        loaded = load_corrections()
        assert loaded[0]["correction_type"] == "brand_swap"

    def test_correction_type_brand_added(self):
        save_correction({
            "serial": "001",
            "ai_result": {"brands": ["MEVIUS"], "skus": []},
            "corrected_result": {"brands": ["MEVIUS", "ARA"], "skus": []},
            "notes": "",
        })
        loaded = load_corrections()
        assert loaded[0]["correction_type"] == "brand_added"

    def test_correction_type_brand_removed(self):
        save_correction({
            "serial": "001",
            "ai_result": {"brands": ["MEVIUS", "ARA"], "skus": []},
            "corrected_result": {"brands": ["MEVIUS"], "skus": []},
            "notes": "",
        })
        loaded = load_corrections()
        assert loaded[0]["correction_type"] == "brand_removed"

    def test_correction_type_sku_corrected(self):
        save_correction({
            "serial": "001",
            "ai_result": {"brands": ["MEVIUS"], "skus": ["MEVIUS ORIGINAL"]},
            "corrected_result": {"brands": ["MEVIUS"], "skus": ["MEVIUS SKY BLUE"]},
            "notes": "",
        })
        loaded = load_corrections()
        assert loaded[0]["correction_type"] == "sku_corrected"


# ═══════════════════════════════════════════════════════════════════════════════
# image_analyzer.py tests
# ═══════════════════════════════════════════════════════════════════════════════

from image_analyzer import _parse_response, _resize_image, get_available_models, MODEL_REGISTRY, MAX_IMAGE_SIZE


class TestImageAnalyzer:
    def test_parse_response_raw_json(self):
        raw = '{"brands_found": [{"brand": "MEVIUS", "skus": ["MEVIUS ORIGINAL"]}], "brand_count": 1, "confidence": "high"}'
        result = _parse_response(raw)
        assert result["brand_count"] == 1
        assert result["brands_found"][0]["brand"] == "MEVIUS"

    def test_parse_response_markdown_wrapped(self):
        md = '```json\n{"brands_found": [], "brand_count": 0, "confidence": "high"}\n```'
        result = _parse_response(md)
        assert result["brand_count"] == 0
        assert result["confidence"] == "high"

    def test_parse_response_invalid_json(self):
        invalid = "This is not JSON at all"
        result = _parse_response(invalid)
        assert result["brands_found"] == []
        assert result["brand_count"] == 0
        assert result["confidence"] == "low"
        assert "Could not parse" in result["notes"]

    def test_parse_response_json_with_text_around(self):
        mixed = 'Here is the analysis:\n{"brands_found": [{"brand": "ARA"}], "brand_count": 1, "confidence": "medium"}\nDone.'
        result = _parse_response(mixed)
        assert result["brand_count"] == 1

    def test_resize_image_keeps_under_max_size(self):
        # Create a large test image
        from PIL import Image
        import io
        img = Image.new("RGB", (4000, 3000), color="red")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        large_data = buf.getvalue()

        resized_data, media_type = _resize_image(large_data, "image/jpeg")
        assert len(resized_data) <= MAX_IMAGE_SIZE
        assert media_type == "image/jpeg"

    def test_resize_image_rgba_conversion(self):
        from PIL import Image
        import io
        # RGBA image (e.g. PNG with transparency)
        img = Image.new("RGBA", (200, 200), color=(255, 0, 0, 128))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_data = buf.getvalue()

        resized_data, media_type = _resize_image(png_data, "image/png")
        # Should convert to JPEG
        assert media_type == "image/jpeg"
        # Should be valid image data
        result_img = Image.open(io.BytesIO(resized_data))
        assert result_img.mode == "RGB"

    def test_get_available_models_returns_nonempty_list(self):
        models = get_available_models()
        assert len(models) > 0
        assert isinstance(models, list)

    def test_model_registry_has_valid_entries(self):
        for name, (provider, model_id) in MODEL_REGISTRY.items():
            assert provider in ("claude", "gemini", "fireworks"), f"Invalid provider for {name}: {provider}"
            assert model_id, f"Empty model_id for {name}"
            assert isinstance(name, str)
            assert isinstance(model_id, str)

    def test_model_registry_matches_available_models(self):
        models = get_available_models()
        assert set(models) == set(MODEL_REGISTRY.keys())


# ═══════════════════════════════════════════════════════════════════════════════
# rate_limiter.py tests
# ═══════════════════════════════════════════════════════════════════════════════

from rate_limiter import RateLimiter, RATE_LIMITS, DEFAULT_RPM


class TestRateLimiter:
    def test_enforces_minimum_interval(self):
        limiter = RateLimiter("gemini-2.5-flash")
        start = time.time()
        limiter.wait()
        limiter.wait()
        elapsed = time.time() - start
        # Should have waited at least one interval
        assert elapsed >= limiter.interval * 0.9  # allow 10% tolerance

    def test_get_safe_workers_returns_reasonable_values(self):
        for model_name in RATE_LIMITS:
            limiter = RateLimiter(model_name)
            workers = limiter.get_safe_workers()
            assert workers >= 1
            assert workers <= limiter.rpm  # can't have more workers than RPM

    def test_get_safe_workers_default_model(self):
        limiter = RateLimiter("unknown-model")
        assert limiter.rpm == DEFAULT_RPM
        workers = limiter.get_safe_workers()
        assert workers >= 1

    def test_rate_limiter_interval_calculation(self):
        limiter = RateLimiter("gemini-2.5-pro")
        expected_interval = 60.0 / RATE_LIMITS["gemini-2.5-pro"]
        assert abs(limiter.interval - expected_interval) < 0.001


# ═══════════════════════════════════════════════════════════════════════════════
# validation.py tests
# ═══════════════════════════════════════════════════════════════════════════════

from validation import validate_image_url, estimate_processing_time


class TestValidation:
    def test_validate_image_url_accepts_valid_https(self):
        assert validate_image_url("https://example.com/image.jpg") is True

    def test_validate_image_url_accepts_valid_http(self):
        assert validate_image_url("http://example.com/image.png") is True

    def test_validate_image_url_rejects_empty(self):
        assert validate_image_url("") is False

    def test_validate_image_url_rejects_none(self):
        assert validate_image_url(None) is False

    def test_validate_image_url_rejects_non_url(self):
        assert validate_image_url("not-a-url") is False
        assert validate_image_url("ftp://example.com/img.jpg") is False

    def test_validate_image_url_rejects_non_string(self):
        assert validate_image_url(12345) is False

    def test_estimate_processing_time_returns_reasonable_estimates(self):
        result = estimate_processing_time(10, model="gemini-2.5-pro")
        assert result["estimated_seconds"] > 0
        assert result["workers"] >= 1
        assert result["rpm"] > 0
        assert "display" in result
        assert "per_image_seconds" in result

    def test_estimate_processing_time_more_images_takes_longer(self):
        small = estimate_processing_time(5, model="gemini-2.5-pro")
        large = estimate_processing_time(50, model="gemini-2.5-pro")
        assert large["estimated_seconds"] > small["estimated_seconds"]

    def test_estimate_processing_time_enhancements_add_overhead(self):
        without = estimate_processing_time(10, model="gemini-2.5-pro", enhancements_enabled=False)
        with_enh = estimate_processing_time(10, model="gemini-2.5-pro", enhancements_enabled=True)
        assert with_enh["estimated_seconds"] > without["estimated_seconds"]


# ═══════════════════════════════════════════════════════════════════════════════
# retry.py tests
# ═══════════════════════════════════════════════════════════════════════════════

from retry import retry_with_backoff


class TestRetry:
    def test_retry_with_backoff_retries_on_failure(self):
        call_count = 0

        @retry_with_backoff(max_retries=2, base_delay=0.01, max_delay=0.05)
        def flaky_fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("temporary failure")
            return "success"

        result = flaky_fn()
        assert result == "success"
        assert call_count == 3

    def test_retry_does_not_retry_on_auth_errors_401(self):
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def auth_fail():
            nonlocal call_count
            call_count += 1
            raise Exception("401 Unauthorized")

        with pytest.raises(Exception, match="401"):
            auth_fail()
        assert call_count == 1  # should not retry

    def test_retry_does_not_retry_on_auth_errors_403(self):
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def auth_fail():
            nonlocal call_count
            call_count += 1
            raise Exception("403 Forbidden")

        with pytest.raises(Exception, match="403"):
            auth_fail()
        assert call_count == 1

    def test_retry_raises_after_max_retries(self):
        @retry_with_backoff(max_retries=2, base_delay=0.01, max_delay=0.02)
        def always_fail():
            raise ConnectionError("always fails")

        with pytest.raises(ConnectionError):
            always_fail()

    def test_retry_succeeds_immediately_without_retry(self):
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def immediate_success():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = immediate_success()
        assert result == "ok"
        assert call_count == 1
