"""Model input serialization for pilot examples."""


def serialize_model_input(endpoint_contract: dict, request: dict) -> str:
    constraints = "\n".join(
        f"- {constraint['constraint_text']}"
        for constraint in endpoint_contract["constraints"]
    )
    query_params = _format_mapping(request.get("query_params", {}))
    body = request.get("body")
    body_fields = _format_mapping(body) if isinstance(body, dict) else f"- body: {body}"

    return (
        "Endpoint description:\n"
        f"{endpoint_contract['description']}\n\n"
        "Required constraints:\n"
        f"{constraints}\n\n"
        "Request:\n"
        f"Method: {request['method']}\n"
        f"URL: {request['url']}\n"
        f"Authentication: {request['authentication']}\n"
        "Query parameters:\n"
        f"{query_params}\n"
        "Body fields:\n"
        f"{body_fields}"
    )


def _format_mapping(values: dict) -> str:
    if not values:
        return "none"
    return "\n".join(f"- {name}: {value}" for name, value in values.items())
