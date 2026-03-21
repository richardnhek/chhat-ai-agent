"""Official brand and SKU list from CHHAT Excel Sheet 2 (Q12B)."""

# Q12B SKUs — exactly as defined in the client's "List of mother brand + SKUs" sheet
BRANDS_AND_SKUS = {
    "MEVIUS": [
        "MEVIUS ORIGINAL", "MEVIUS SKY BLUE", "MEVIUS OPTION PURPLE",
        "MEVIUS FREEZY DEW", "MEVIUS OPTION PURPLE SUPER SLIMS",
        "MEVIUS KIWAMI", "MEVIUS E-SERIES BLUE", "MEVIUS MINT FLOW",
    ],
    "WINSTON": ["WINSTON NIGHT BLUE", "WINSTON OPTION PURPLE", "WINSTON OPTION BLUE"],
    "ESSE": ["ESSE CHANGE", "ESSE LIGHTS", "ESSE MENTHOL", "ESSE GOLD", "ESSE OTHERS"],
    "FINE": ["FINE RED HARD PACK", "FINE OTHERS"],
    "555": ["555 SPHERE2 VELVETY", "555 ORIGINAL", "555 GOLD", "555 OTHERS"],
    "ARA": ["ARA RED", "ARA GOLD", "ARA MENTHOL", "ARA OTHERS"],
    "LUXURY": ["LUXURY FULL FLAVOUR", "LUXURY MENTHOL", "LUXURY OTHERS"],
    "GOLD SEAL": ["GOLD SEAL MENTHOL COMPACT", "GOLD SEAL MENTHOL KINGSIZE", "GOLD SEAL OTHERS"],
    "MARLBORO": ["MARLBORO RED", "MARLBORO GOLD", "MARLBORO OTHERS"],
    "CAMBO": ["CAMBO CLASSICAL", "CAMBO FF", "CAMBO MENTHOL"],
    "IZA": ["IZA FF", "IZA MENTHOL", "IZA OTHERS"],
    "HERO": ["HERO HARD PACK"],
    "COW BOY": ["COW BOY BLUEBERRY MINT", "COW BOY HARD PACK", "COW BOY MENTHOL", "COW BOY OTHERS"],
    "COCO PALM": ["COCO PALM HARD PACK", "COCO PALM MENTHOL", "COCO PALM OTHERS"],
    "CROWN": ["CROWN"],
    "LAPIN": ["LAPIN FF", "LAPIN MENTHOL"],
    "ORIS": ["ORIS PULSE BLUE", "ORIS ICE PLUS", "ORIS SILVER", "ORIS OTHERS"],
    "JET": ["JET"],
    "L&M": ["L&M"],
    "DJARUM": ["DJARUM"],
    "LIBERATION": ["LIBERATION"],
    "MODERN": ["MODERN"],
    "MOND": ["MOND"],
    "NATIONAL": ["NATIONAL"],
    "CHUNGHWA": ["CHUNGHWA"],
    "SHUANGXI": ["SHUANGXI"],
    "YUN YAN": ["YUN YAN"],
    "CHINESE BRAND": ["CHINESE BRANDS"],
    "OTHERS": ["OTHERS"],
}

BRAND_KHMER = {
    "MEVIUS": "ម៉ៃសេវែន / មេវៀស",
    "WINSTON": "វីនស្តុន",
    "ESSE": "អេសសេ",
    "FINE": "ហ្វីន",
    "555": "បារី​​ 555",
    "ARA": "សេក",
    "LUXURY": "លុចសារី",
    "GOLD SEAL": "ហ្គោលសៀល",
    "MARLBORO": "ម៉ាបូរ៉ូ",
    "CAMBO": "ខេមបូ",
    "IZA": "អ៊ីសា",
    "HERO": "ហេរ៉ូ",
    "COW BOY": "ខោប៊យ",
    "COCO PALM": "ដើមដូង",
    "CROWN": "ក្រោន",
    "LAPIN": "ឡាពីន / ទន្សាយ",
    "ORIS": "អូរីស",
    "JET": "ជែត",
    "L&M": "អិល អ៊ែន អឹម",
    "DJARUM": "ចារ៉ាម",
    "LIBERATION": "រំដោះ",
    "MODERN": "ម៉ូឌឺន",
    "MOND": "ម៉ន់",
    "NATIONAL": "ជាតិ",
    "CHUNGHWA": "ឆុងវ៉ា",
    "SHUANGXI": "ស៊ុងស៊ីង",
    "YUN YAN": "យ័នយែន",
    "CHINESE BRAND": "ម៉ាកចិន",
    "OTHERS": "ម៉ាកផ្សេងៗ",
}

def get_brand_list_for_prompt() -> str:
    """Format the brand list for the AI prompt."""
    lines = []
    for brand, skus in BRANDS_AND_SKUS.items():
        sku_str = ", ".join(skus)
        lines.append(f"- {brand}: SKUs = [{sku_str}]")
    return "\n".join(lines)

def format_q12a(brands: list[str]) -> str:
    """Format brands into Q12A output format: BRAND_Khmer | BRAND_Khmer"""
    parts = []
    for b in brands:
        khmer = BRAND_KHMER.get(b, "")
        parts.append(f"{b}_{khmer}" if khmer else b)
    return " | ".join(parts)

def format_q12b(skus: list[str]) -> str:
    """Format SKUs into Q12B output format: SKU1 | SKU2"""
    return " | ".join(skus)
