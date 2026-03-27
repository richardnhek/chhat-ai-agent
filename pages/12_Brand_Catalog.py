"""
Brand & SKU Manager — manage the CHHAT brand catalog from the UI.
"""

import os
import json
from pathlib import Path

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from auth import check_auth
from brand_catalog import (
    load_brand_catalog,
    save_brand_catalog,
    get_reference_image_counts,
    get_annotation_counts,
)

load_dotenv()

st.set_page_config(page_title="Brand Catalog", page_icon="📋", layout="wide")

if not check_auth():
    st.stop()

# ── Styles ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0; }
    .sub-header { font-size: 1.05rem; color: #6c757d; margin-bottom: 1.8rem; font-style: italic; }
    .tier-1 { background: #d4edda; color: #155724; padding: 0.2rem 0.6rem; border-radius: 6px; font-weight: 600; font-size: 0.8rem; }
    .tier-2 { background: #fff3cd; color: #856404; padding: 0.2rem 0.6rem; border-radius: 6px; font-weight: 600; font-size: 0.8rem; }
    .tier-3 { background: #f8d7da; color: #721c24; padding: 0.2rem 0.6rem; border-radius: 6px; font-weight: 600; font-size: 0.8rem; }
    .brand-tag { display: inline-block; background: #4472C4; color: white; padding: 0.2rem 0.6rem; border-radius: 20px; font-size: 0.8rem; margin: 0.1rem; }
    .sku-count { font-size: 0.85rem; color: #6c757d; }
    .stat-pill { display: inline-block; background: #e9ecef; padding: 0.15rem 0.5rem; border-radius: 10px; font-size: 0.75rem; margin: 0.1rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">Brand & SKU Manager</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Manage the CHHAT brand catalog without editing code</p>', unsafe_allow_html=True)

# ── Load catalog ──────────────────────────────────────────────────────────
catalog = load_brand_catalog()
ref_counts = get_reference_image_counts()
ann_counts = get_annotation_counts()

# ── Tabs ──────────────────────────────────────────────────────────────────
tab_overview, tab_add, tab_edit, tab_priority = st.tabs([
    "Brand Overview", "Add Brand", "Edit Brand / SKUs", "Brand Priority"
])

# =====================================================================
# TAB 1: Brand Overview Table
# =====================================================================
with tab_overview:
    st.markdown("### All Brands")
    st.caption("Toggle brands active/inactive. Inactive brands are excluded from training and analysis.")

    brands = catalog["brands"]
    rows = []
    for brand_name, info in brands.items():
        rows.append({
            "Brand": brand_name,
            "Khmer": info.get("khmer", ""),
            "SKUs": len(info.get("skus", [])),
            "Tier": info.get("tier", 2),
            "Active": info.get("active", True),
            "Annotations": ann_counts.get(brand_name, 0),
            "Priority": info.get("priority", 99),
        })

    if not rows:
        st.info("No brands in the catalog yet. Use the 'Add Brand' tab to get started.")
    else:
        # Sort by priority
        rows.sort(key=lambda r: r["Priority"])

        # Display with toggle switches
        changes_made = False

        # Header
        hcols = st.columns([2.5, 2.5, 0.8, 0.8, 1, 1.2])
        hcols[0].markdown("**Brand**")
        hcols[1].markdown("**Khmer Name**")
        hcols[2].markdown("**SKUs**")
        hcols[3].markdown("**Tier**")
        hcols[4].markdown("**Annotations**")
        hcols[5].markdown("**Active**")

        st.markdown("<hr style='margin: 0.3rem 0; border-color: #e0e0e0;'>", unsafe_allow_html=True)

        for i, row in enumerate(rows):
            cols = st.columns([2.5, 2.5, 0.8, 0.8, 1, 1.2])
            cols[0].markdown(f"**{row['Brand']}**")
            cols[1].markdown(row["Khmer"] if row["Khmer"] else "-")
            cols[2].markdown(str(row["SKUs"]))

            tier = row["Tier"]
            tier_cls = f"tier-{tier}" if tier in (1, 2, 3) else "tier-2"
            tier_labels = {1: "Focus", 2: "Important", 3: "Rare"}
            cols[3].markdown(
                f'<span class="{tier_cls}">{tier_labels.get(tier, "T" + str(tier))}</span>',
                unsafe_allow_html=True,
            )
            cols[4].markdown(str(row["Annotations"]))

            new_active = cols[5].toggle(
                "Active",
                value=row["Active"],
                key=f"active_{row['Brand']}_{i}",
                label_visibility="collapsed",
            )
            if new_active != row["Active"]:
                catalog["brands"][row["Brand"]]["active"] = new_active
                changes_made = True

        if changes_made:
            save_brand_catalog(catalog)
            st.success("Brand status updated and saved.")
            st.rerun()

        # Summary metrics
        st.markdown("---")
        mcols = st.columns(4)
        total = len(brands)
        active = sum(1 for b in brands.values() if b.get("active", True))
        total_skus = sum(len(b.get("skus", [])) for b in brands.values())
        total_anns = sum(ann_counts.values())
        mcols[0].metric("Total Brands", total)
        mcols[1].metric("Active Brands", active)
        mcols[2].metric("Total SKUs", total_skus)
        mcols[3].metric("Total Annotations", total_anns)

# =====================================================================
# TAB 2: Add Brand
# =====================================================================
with tab_add:
    st.markdown("### Add New Brand")

    with st.form("add_brand_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            new_brand_name = st.text_input("Brand Name (English)", placeholder="e.g. LUCKY STRIKE")
        with col2:
            new_brand_khmer = st.text_input("Khmer Name", placeholder="e.g. ឡាកគីស្រ្តាយ")

        new_skus_text = st.text_area(
            "SKUs (comma-separated or one per line)",
            placeholder="LUCKY STRIKE RED, LUCKY STRIKE BLUE\nor one per line",
            height=120,
        )

        new_ref_images = st.file_uploader(
            "Reference Images (optional)",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
        )

        submitted = st.form_submit_button("Add Brand", type="primary", use_container_width=True)

        if submitted:
            name = new_brand_name.strip().upper()
            if not name:
                st.error("Brand name is required.")
            elif name in catalog["brands"]:
                st.error(f"Brand '{name}' already exists. Use the Edit tab to modify it.")
            else:
                # Parse SKUs
                raw = new_skus_text.replace("\n", ",")
                skus = [s.strip().upper() for s in raw.split(",") if s.strip()]
                if not skus:
                    skus = [name]  # default SKU = brand name

                # Add to catalog
                catalog["brands"][name] = {
                    "skus": skus,
                    "khmer": new_brand_khmer.strip(),
                    "active": True,
                    "tier": 2,
                    "priority": len(catalog["brands"]) + 1,
                }
                save_brand_catalog(catalog)

                # Save reference images
                if new_ref_images:
                    ref_dir = Path(__file__).parent.parent / "reference_images"
                    ref_dir.mkdir(exist_ok=True)
                    # Find next available image number
                    existing = list(ref_dir.glob("image*.*"))
                    max_num = 0
                    for f in existing:
                        stem = f.stem.replace("image", "")
                        try:
                            max_num = max(max_num, int(stem))
                        except ValueError:
                            pass
                    for idx, img_file in enumerate(new_ref_images, start=max_num + 1):
                        ext = Path(img_file.name).suffix
                        dest = ref_dir / f"image{idx}{ext}"
                        dest.write_bytes(img_file.getvalue())

                st.success(f"Brand '{name}' added with {len(skus)} SKU(s).")
                st.rerun()

# =====================================================================
# TAB 3: Edit Brand / SKUs
# =====================================================================
with tab_edit:
    st.markdown("### Edit Brand / SKUs")

    brand_names = sorted(catalog["brands"].keys())
    if not brand_names:
        st.info("No brands to edit. Add a brand first.")
    else:
        selected_brand = st.selectbox("Select Brand", brand_names, key="edit_brand_select")
        brand_data = catalog["brands"][selected_brand]

        st.markdown(f"**Khmer:** {brand_data.get('khmer', '-')}")
        st.markdown(f"**Status:** {'Active' if brand_data.get('active', True) else 'Inactive'}")

        st.markdown("#### Current SKUs")
        current_skus = brand_data.get("skus", [])

        # Show current SKUs with remove buttons
        skus_to_remove = set()
        if current_skus:
            for j, sku in enumerate(current_skus):
                sc1, sc2 = st.columns([4, 1])
                sc1.markdown(f"`{sku}`")
                if sc2.button("Remove", key=f"rm_sku_{selected_brand}_{j}", type="secondary"):
                    skus_to_remove.add(sku)

            if skus_to_remove:
                remaining = [s for s in current_skus if s not in skus_to_remove]
                catalog["brands"][selected_brand]["skus"] = remaining
                save_brand_catalog(catalog)
                st.success(f"Removed {len(skus_to_remove)} SKU(s).")
                st.rerun()
        else:
            st.caption("No SKUs defined.")

        st.markdown("#### Add SKUs")
        with st.form("edit_skus_form", clear_on_submit=True):
            add_skus_text = st.text_area(
                "New SKUs (comma-separated or one per line)",
                height=80,
                key="edit_add_skus",
            )

            edit_khmer = st.text_input(
                "Update Khmer Name (leave blank to keep current)",
                value="",
                key="edit_khmer_input",
            )

            add_ref_images = st.file_uploader(
                "Upload Additional Reference Images",
                type=["jpg", "jpeg", "png", "webp"],
                accept_multiple_files=True,
                key="edit_ref_images",
            )

            save_btn = st.form_submit_button("Save Changes", type="primary", use_container_width=True)

            if save_btn:
                updated = False

                # Add new SKUs
                if add_skus_text.strip():
                    raw = add_skus_text.replace("\n", ",")
                    new_skus = [s.strip().upper() for s in raw.split(",") if s.strip()]
                    existing = set(catalog["brands"][selected_brand]["skus"])
                    added = [s for s in new_skus if s not in existing]
                    catalog["brands"][selected_brand]["skus"].extend(added)
                    if added:
                        updated = True
                        st.info(f"Added {len(added)} new SKU(s): {', '.join(added)}")

                # Update Khmer
                if edit_khmer.strip():
                    catalog["brands"][selected_brand]["khmer"] = edit_khmer.strip()
                    updated = True

                # Save reference images
                if add_ref_images:
                    ref_dir = Path(__file__).parent.parent / "reference_images"
                    ref_dir.mkdir(exist_ok=True)
                    existing_files = list(ref_dir.glob("image*.*"))
                    max_num = 0
                    for f in existing_files:
                        stem = f.stem.replace("image", "")
                        try:
                            max_num = max(max_num, int(stem))
                        except ValueError:
                            pass
                    for idx, img_file in enumerate(add_ref_images, start=max_num + 1):
                        ext = Path(img_file.name).suffix
                        dest = ref_dir / f"image{idx}{ext}"
                        dest.write_bytes(img_file.getvalue())
                    updated = True
                    st.info(f"Saved {len(add_ref_images)} reference image(s).")

                if updated:
                    save_brand_catalog(catalog)
                    st.success("Changes saved.")
                    st.rerun()
                else:
                    st.warning("No changes to save.")

        # Delete brand
        st.markdown("---")
        st.markdown("#### Danger Zone")
        with st.expander("Delete this brand"):
            st.warning(f"This will permanently remove **{selected_brand}** from the catalog.")
            if st.button(f"Delete {selected_brand}", type="primary", key="delete_brand_btn"):
                del catalog["brands"][selected_brand]
                save_brand_catalog(catalog)
                st.success(f"Brand '{selected_brand}' deleted.")
                st.rerun()

# =====================================================================
# TAB 4: Brand Priority / Tier
# =====================================================================
with tab_priority:
    st.markdown("### Brand Priority & Tier Assignment")
    st.caption(
        "Assign tiers to indicate brand importance. "
        "Tier 1 = Focus brands, Tier 2 = Important, Tier 3 = Rare/low priority."
    )

    brand_names_sorted = sorted(
        catalog["brands"].keys(),
        key=lambda b: catalog["brands"][b].get("priority", 99),
    )

    if not brand_names_sorted:
        st.info("No brands in the catalog.")
    else:
        tier_changes = False

        # Header
        hcols = st.columns([2.5, 2, 2, 1.5])
        hcols[0].markdown("**Brand**")
        hcols[1].markdown("**Current Tier**")
        hcols[2].markdown("**Set Tier**")
        hcols[3].markdown("**Priority #**")

        st.markdown("<hr style='margin: 0.3rem 0; border-color: #e0e0e0;'>", unsafe_allow_html=True)

        tier_options = {1: "Tier 1 — Focus", 2: "Tier 2 — Important", 3: "Tier 3 — Rare"}

        for k, brand_name in enumerate(brand_names_sorted):
            info = catalog["brands"][brand_name]
            cols = st.columns([2.5, 2, 2, 1.5])

            cols[0].markdown(f"**{brand_name}**")

            current_tier = info.get("tier", 2)
            tier_cls = f"tier-{current_tier}" if current_tier in (1, 2, 3) else "tier-2"
            tier_label = {1: "Focus", 2: "Important", 3: "Rare"}.get(current_tier, str(current_tier))
            cols[1].markdown(f'<span class="{tier_cls}">{tier_label}</span>', unsafe_allow_html=True)

            new_tier = cols[2].selectbox(
                "Tier",
                options=[1, 2, 3],
                format_func=lambda x: tier_options[x],
                index=current_tier - 1,
                key=f"tier_{brand_name}_{k}",
                label_visibility="collapsed",
            )

            new_priority = cols[3].number_input(
                "Priority",
                min_value=1,
                max_value=len(brand_names_sorted),
                value=info.get("priority", k + 1),
                key=f"prio_{brand_name}_{k}",
                label_visibility="collapsed",
            )

            if new_tier != current_tier:
                catalog["brands"][brand_name]["tier"] = new_tier
                tier_changes = True
            if new_priority != info.get("priority", k + 1):
                catalog["brands"][brand_name]["priority"] = new_priority
                tier_changes = True

        st.markdown("---")
        if st.button("Save Priority & Tier Changes", type="primary", use_container_width=True):
            save_brand_catalog(catalog)
            st.success("Priority and tier changes saved.")
            st.rerun()

        # Summary
        st.markdown("---")
        st.markdown("#### Tier Summary")
        tier_summary = {1: [], 2: [], 3: []}
        for bn, bi in catalog["brands"].items():
            t = bi.get("tier", 2)
            if t in tier_summary:
                tier_summary[t].append(bn)

        tcols = st.columns(3)
        for i, (tier_num, tier_brands) in enumerate(tier_summary.items()):
            with tcols[i]:
                label = {1: "Tier 1 — Focus", 2: "Tier 2 — Important", 3: "Tier 3 — Rare"}[tier_num]
                st.markdown(f"**{label}** ({len(tier_brands)})")
                for b in sorted(tier_brands):
                    st.markdown(f"- {b}")
