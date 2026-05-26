"""Version 1 split and endpoint-family planning."""

from semantic_api_diagnosis.contracts.endpoint_contract import CONTRACTS

SUPPORTED_SPLITS = ["pilot", "train", "validation", "seen_test", "unseen_family_test"]
SUPPORTED_FAMILY_MODES = ["implemented", "seen", "unseen", "all"]

IMPLEMENTED_ENDPOINT_FAMILIES = tuple(CONTRACTS.keys())

PLANNED_SEEN_ENDPOINT_FAMILIES = (
    "refunds_orders",
    "booking_reservation",
    "user_permissions",
    "inventory_product_search",
    "payments_invoices",
)

PLANNED_UNSEEN_ENDPOINT_FAMILIES = (
    "healthcare_appointments",
    "course_registration",
    "ticket_support_workflow",
    "shipping_returns",
    "subscription_plan_changes",
)

RECOMMENDED_V1_TARGET_SIZES = {
    "train": "3000-4000",
    "validation": "500",
    "seen_test": "700-1000",
    "unseen_family_test": "700-1000",
}


def implemented_seen_families() -> tuple[str, ...]:
    return tuple(
        family
        for family in PLANNED_SEEN_ENDPOINT_FAMILIES
        if family in IMPLEMENTED_ENDPOINT_FAMILIES
    )


def implemented_unseen_families() -> tuple[str, ...]:
    return tuple(
        family
        for family in PLANNED_UNSEEN_ENDPOINT_FAMILIES
        if family in IMPLEMENTED_ENDPOINT_FAMILIES
    )


def planned_not_implemented_seen_families() -> tuple[str, ...]:
    return tuple(
        family
        for family in PLANNED_SEEN_ENDPOINT_FAMILIES
        if family not in IMPLEMENTED_ENDPOINT_FAMILIES
    )


def planned_not_implemented_unseen_families() -> tuple[str, ...]:
    return tuple(
        family
        for family in PLANNED_UNSEEN_ENDPOINT_FAMILIES
        if family not in IMPLEMENTED_ENDPOINT_FAMILIES
    )


def resolve_families(split: str, family_mode: str) -> tuple[str, ...]:
    if split not in SUPPORTED_SPLITS:
        raise ValueError(f"Unsupported split '{split}'. Supported splits: {', '.join(SUPPORTED_SPLITS)}")
    if family_mode not in SUPPORTED_FAMILY_MODES:
        raise ValueError(
            f"Unsupported family mode '{family_mode}'. Supported modes: {', '.join(SUPPORTED_FAMILY_MODES)}"
        )
    if split == "unseen_family_test" and family_mode in {"unseen", "all"}:
        families = implemented_unseen_families()
        if not families:
            raise ValueError(
                "unseen_family_test requested, but no unseen endpoint families are implemented yet. "
                "Planned unseen families: " + ", ".join(PLANNED_UNSEEN_ENDPOINT_FAMILIES)
            )
        return families
    if family_mode == "unseen":
        families = implemented_unseen_families()
        if not families:
            raise ValueError(
                "family-mode unseen requested, but no unseen endpoint families are implemented yet. "
                "Planned unseen families: " + ", ".join(PLANNED_UNSEEN_ENDPOINT_FAMILIES)
            )
        return families
    if family_mode == "implemented" or split == "pilot":
        return IMPLEMENTED_ENDPOINT_FAMILIES
    if family_mode == "seen" or split in {"train", "validation", "seen_test"}:
        families = implemented_seen_families()
        if not families:
            raise ValueError("No implemented seen endpoint families are available.")
        return families
    return IMPLEMENTED_ENDPOINT_FAMILIES


def dataset_plan_text() -> str:
    unseen_status = (
        "- unseen_family_test generation is available for implemented unseen families"
        if implemented_unseen_families()
        else "- unseen_family_test generation is blocked until at least one unseen family is implemented"
    )
    lines = [
        "Semantic API Diagnosis Dataset Plan",
        "",
        "Implemented endpoint families:",
        *_bullet_lines(IMPLEMENTED_ENDPOINT_FAMILIES),
        "",
        "Planned seen endpoint families:",
        *_status_lines(PLANNED_SEEN_ENDPOINT_FAMILIES),
        "",
        "Planned unseen endpoint families:",
        *_status_lines(PLANNED_UNSEEN_ENDPOINT_FAMILIES),
        "",
        "Supported splits:",
        *_bullet_lines(SUPPORTED_SPLITS),
        "",
        "Recommended Version 1 target sizes:",
        *[f"- {split}: {size}" for split, size in RECOMMENDED_V1_TARGET_SIZES.items()],
        "",
        "Current implementation status:",
        f"- implemented seen families: {len(implemented_seen_families())}/{len(PLANNED_SEEN_ENDPOINT_FAMILIES)}",
        f"- implemented unseen families: {len(implemented_unseen_families())}/{len(PLANNED_UNSEEN_ENDPOINT_FAMILIES)}",
        unseen_status,
    ]
    return "\n".join(lines)


def _bullet_lines(values: tuple[str, ...] | list[str]) -> list[str]:
    return [f"- {value}" for value in values]


def _status_lines(values: tuple[str, ...]) -> list[str]:
    return [
        f"- {value}: {'implemented' if value in IMPLEMENTED_ENDPOINT_FAMILIES else 'planned'}"
        for value in values
    ]
