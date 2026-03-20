"""Official brand list from the KH Census Brand Book."""

BRANDS_AND_SKUS = {
    "MEVIUS": [
        "MEVIUS ORIGINAL", "MEVIUS SKY BLUE", "MEVIUS OPTION PURPLE",
        "MEVIUS FREEZY DEW", "MEVIUS OPTION PURPLE SUPER SLIMS",
        "MEVIUS KIWAMI", "MEVIUS E-SERIES BLUE", "MEVIUS MINT FLOW",
    ],
    "WINSTON": [
        "WINSTON NIGHT BLUE", "WINSTON OPTION PURPLE", "WINSTON OPTION BLUE",
    ],
    "ESSE": [
        "ESSE CHANGE", "ESSE LIGHTS", "ESSE MENTHOL", "ESSE GOLD",
        "ESSE CHANGE CAFE", "ESSE BLACK", "ESSE RED",
        "ESSE CHANGE COOLIPS SWEET APPLE", "ESSE CHANGE DOUBLE",
        "ESSE CHANGE DOUBLE SHOT", "ESSE CHANGE FROZEN PEACH MOJITO",
        "ESSE CHANGE HIMALAYA", "ESSE CHANGE MANGO",
        "ESSE CHANGE SHOOTING RED COLA", "ESSE CHANGE STRAWBERRY",
        "ESSE GREEN", "ESSE IT'S BUBBLE PURPLE", "ESSE IT'S DEEP BROWN",
    ],
    "FINE": [
        "FINE RED HARD PACK", "FINE GOLD", "FINE MENTHOL", "FINE CHARCOAL FILTER",
    ],
    "555": [
        "555 SPHERE2 VELVETY", "555 ORIGINAL", "555 GOLD",
        "555 REFINED CHARCOAL FILTER", "555 SWITCH", "555 BERRY BOOST",
        "555 PRESTIGE", "555 SLIM", "555 SPHERE2 SPARKY",
    ],
    "ARA": [
        "ARA RED", "ARA GOLD", "ARA MENTHOL", "ARA COOL", "ARA TROPICAL",
        "ARA NEXT", "ARA ORIGINAL", "ARA PREMIER",
    ],
    "LUXURY": [
        "LUXURY FULL FLAVOUR", "LUXURY MENTHOL",
        "LUXURY SS DOUBLE CAPSULE", "LUXURY BLUEBERRY MINT OPTION",
        "LUXURY ORANGE OPTION", "LUXURY BLUEBERRY OPTION",
        "LUXURY LIGHTS", "LUXURY MENTHOL OPTION",
        "LUXURY PREMIUM OPTION BLUEBERRY", "LUXURY SPECIAL BLEND",
    ],
    "GOLD SEAL": [
        "GOLD SEAL MENTHOL COMPACT", "GOLD SEAL MENTHOL KING SIZE",
        "GOLD SEAL FULL FLAVOR", "GOLD SEAL GOLD MIDI SLIMS",
        "GOLD SEAL CLASSIC RED", "GOLD SEAL SPECIAL GOLD",
    ],
    "MARLBORO": [
        "MARLBORO RED", "MARLBORO GOLD", "MARLBORO MENTHOL",
        "MARLBORO ICE BLAST", "MARLBORO GOLD ADVANCE",
        "MARLBORO SPLASH", "MARLBORO VISTA FOREST", "MARLBORO VISTA ICE BLAST",
    ],
    "CAMBO": ["CAMBO CLASSICAL", "CAMBO MENTHOL", "CAMBO FF"],
    "IZA": ["IZA FF", "IZA MENTHOL", "IZA LIGHTS", "IZA DOUBLE BURST"],
    "HERO": ["HERO HARD PACK"],
    "COW BOY": [
        "COW BOY BLUEBERRY MINT", "COW BOY HARD PACK", "COW BOY MENTHOL",
        "COW BOY LIGHTS", "COW BOY SUPER SLIMS",
    ],
    "COCO PALM": ["COCO PALM HARD PACK", "COCO PALM MENTHOL", "COCO PALM GOLD"],
    "CROWN": ["CROWN"],
    "LAPIN": ["LAPIN FF", "LAPIN MENTHOL"],
    "ORIS": [
        "ORIS PULSE BLUE", "ORIS ICE PLUS", "ORIS SILVER",
        "ORIS PULSE", "ORIS MENTHOL", "ORIS AZURE BLUE", "ORIS BLACK",
        "ORIS FINE RED", "ORIS INTENSE BLACK CURRANT", "ORIS INTENSE DEEP MIX",
        "ORIS INTENSE GUAVA", "ORIS INTENSE PURPLE FIZZ",
        "ORIS INTENSE SUMMER FIZZ", "ORIS INTENSE TROPICAL DEW",
        "ORIS PULSE APPLEMINT ORANGE", "ORIS PULSE MENTHOL ORANGE",
        "ORIS PULSE SUPER SLIMS STRAWBERRY FUSION", "ORIS RED",
        "ORIS SLIMS CHOCOLATE", "ORIS SLIMS GOLD", "ORIS SLIMS STRAWBERRY",
        "ORIS TWIN SENSE BERRY MIX",
    ],
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
    "CHINESE BRAND": [
        "DIAMOND HEHUA", "DOUBLE HAPPINESS", "HARMONIZATION",
        "HOMATA TOBACCO GROUP", "UANGHELOU", "LIGUN VIRGINIA",
    ],
}

# Khmer translations for brands (for Q12A output format)
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
