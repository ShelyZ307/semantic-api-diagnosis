"""Endpoint contract definitions for the pilot dataset."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FieldSpec:
    name: str
    type_name: str
    required: bool = True
    description: str = ""


@dataclass(frozen=True)
class EndpointContract:
    endpoint_name: str
    description: str
    description_variants: list[str]
    method: str
    url_template: str
    auth_required: bool
    required_fields: dict[str, str]
    optional_fields: dict[str, str] = field(default_factory=dict)
    constraints: list[dict[str, Any]] = field(default_factory=list)

    def to_dataset_dict(self) -> dict[str, Any]:
        return {
            "endpoint_name": self.endpoint_name,
            "description": self.description,
            "method": self.method,
            "url_template": self.url_template,
            "auth_required": self.auth_required,
            "required_fields": self.required_fields,
            "optional_fields": self.optional_fields,
            "constraints": self.constraints,
        }


CONTRACTS: dict[str, EndpointContract] = {
    "refunds_orders": EndpointContract(
        endpoint_name="create_order_refund",
        description="Create a refund request against an existing paid order.",
        description_variants=[
            "Create a refund request against an existing paid order.",
            "Submit a customer refund for an order that has already been paid.",
            "Open a refund transaction for a paid order and its original charge.",
        ],
        method="POST",
        url_template="/orders/{order_id}/refunds",
        auth_required=True,
        required_fields={
            "order_id": "string",
            "refund_amount": "number",
            "original_payment": "number",
            "order_status": "enum:paid|pending|cancelled|refunded",
        },
        optional_fields={"notify_customer": "boolean", "reason": "string"},
        constraints=[
            {
                "constraint_id": "refund_positive",
                "constraint_text": "refund_amount must be positive.",
                "constraint_variants": [
                    "refund_amount must be positive.",
                    "The refund amount must be greater than zero.",
                    "Refund requests require a positive refund_amount value.",
                ],
                "constraint_type": "structural",
                "fields": ["refund_amount"],
                "severity": "medium",
            },
            {
                "constraint_id": "refund_not_over_payment",
                "constraint_text": "The requested refund must not be greater than the original payment.",
                "constraint_variants": [
                    "The requested refund must not be greater than the original payment.",
                    "A refund cannot exceed the amount originally paid by the customer.",
                    "The refund value must be less than or equal to the original charge.",
                ],
                "constraint_type": "semantic_domain",
                "fields": ["refund_amount", "original_payment"],
                "severity": "high",
            },
            {
                "constraint_id": "refund_paid_orders_only",
                "constraint_text": "Refunds are allowed only for orders that have already been paid.",
                "constraint_variants": [
                    "Refunds are allowed only for orders that have already been paid.",
                    "Only paid orders may be refunded.",
                    "An order must be in paid status before a refund can be created.",
                ],
                "constraint_type": "semantic_state",
                "fields": ["order_status"],
                "severity": "high",
            },
            {
                "constraint_id": "refund_url_body_order_match",
                "constraint_text": "The order_id in the body must match the order_id in the URL.",
                "constraint_variants": [
                    "The order_id in the body must match the order_id in the URL.",
                    "The refund body and URL must refer to the same order_id.",
                    "A refund request cannot use different order IDs in the path and body.",
                ],
                "constraint_type": "semantic_cross_field",
                "fields": ["order_id"],
                "severity": "high",
            },
        ],
    ),
    "booking_reservation": EndpointContract(
        endpoint_name="modify_reservation",
        description="Modify dates and guest count for an existing room reservation.",
        description_variants=[
            "Modify dates and guest count for an existing room reservation.",
            "Update an active hotel reservation with revised stay details.",
            "Change reservation dates or occupancy for a booked room.",
        ],
        method="PATCH",
        url_template="/reservations/{reservation_id}",
        auth_required=True,
        required_fields={
            "reservation_id": "string",
            "start_date": "date",
            "end_date": "date",
            "number_of_guests": "integer",
            "room_capacity": "integer",
            "reservation_status": "enum:pending|confirmed|checked_in|cancelled",
        },
        optional_fields={"special_requests": "string"},
        constraints=[
            {
                "constraint_id": "reservation_date_order",
                "constraint_text": "start_date must be before end_date.",
                "constraint_variants": [
                    "start_date must be before end_date.",
                    "The reservation must begin before it ends.",
                    "The check-in date must be earlier than the check-out date.",
                ],
                "constraint_type": "semantic_cross_field",
                "fields": ["start_date", "end_date"],
                "severity": "high",
            },
            {
                "constraint_id": "reservation_capacity",
                "constraint_text": "number_of_guests must not exceed room_capacity.",
                "constraint_variants": [
                    "number_of_guests must not exceed room_capacity.",
                    "The guest count cannot be larger than the room capacity.",
                    "A reservation may not include more guests than the room can hold.",
                ],
                "constraint_type": "semantic_domain",
                "fields": ["number_of_guests", "room_capacity"],
                "severity": "high",
            },
            {
                "constraint_id": "reservation_modifiable_state",
                "constraint_text": "Reservations can only be modified when status is pending or confirmed.",
                "constraint_variants": [
                    "Reservations can only be modified when status is pending or confirmed.",
                    "Only pending or confirmed reservations may be changed.",
                    "A reservation must still be pending or confirmed before updates are accepted.",
                ],
                "constraint_type": "semantic_state",
                "fields": ["reservation_status"],
                "severity": "medium",
            },
        ],
    ),
    "user_permissions": EndpointContract(
        endpoint_name="approve_permission_change",
        description="Approve a user's requested permission or role change.",
        description_variants=[
            "Approve a user's requested permission or role change.",
            "Review and approve a requested update to a user's access role.",
            "Process an approval request for changing user permissions.",
        ],
        method="POST",
        url_template="/users/{user_id}/permissions",
        auth_required=True,
        required_fields={
            "user_id": "string",
            "requested_by": "string",
            "approved_by": "string",
            "current_role": "enum:viewer|editor|manager|admin",
            "target_role": "enum:viewer|editor|manager|admin",
            "approver_role": "enum:viewer|editor|manager|admin",
            "request_status": "enum:pending|approved|rejected",
        },
        optional_fields={"justification": "string"},
        constraints=[
            {
                "constraint_id": "permission_no_self_approval",
                "constraint_text": "A user may not approve their own permission change.",
                "constraint_variants": [
                    "A user may not approve their own permission change.",
                    "The requester and approver must be different users.",
                    "Permission changes cannot be self-approved.",
                ],
                "constraint_type": "semantic_cross_field",
                "fields": ["requested_by", "approved_by"],
                "severity": "high",
            },
            {
                "constraint_id": "permission_approver_role",
                "constraint_text": "Only managers or admins may approve role upgrades.",
                "constraint_variants": [
                    "Only managers or admins may approve role upgrades.",
                    "Role upgrades require approval from a manager or admin.",
                    "A viewer or editor cannot approve a higher target role.",
                ],
                "constraint_type": "semantic_domain",
                "fields": ["current_role", "target_role", "approver_role"],
                "severity": "high",
            },
            {
                "constraint_id": "permission_role_changes_only",
                "constraint_text": "target_role must be different from current_role.",
                "constraint_variants": [
                    "target_role must be different from current_role.",
                    "The requested role must change the user's current role.",
                    "Permission requests should not target the role the user already has.",
                ],
                "constraint_type": "semantic_cross_field",
                "fields": ["current_role", "target_role"],
                "severity": "medium",
            },
            {
                "constraint_id": "permission_pending_requests_only",
                "constraint_text": "Permission changes can only be approved while the request is pending.",
                "constraint_variants": [
                    "Permission changes can only be approved while the request is pending.",
                    "Only pending permission requests may be approved.",
                    "A permission request must still be pending before an approval is accepted.",
                ],
                "constraint_type": "semantic_state",
                "fields": ["request_status"],
                "severity": "high",
            },
        ],
    ),
    "inventory_product_search": EndpointContract(
        endpoint_name="reserve_product_inventory",
        description="Reserve product inventory for a customer order.",
        description_variants=[
            "Reserve product inventory for a customer order.",
            "Create or update a stock reservation for a product.",
            "Hold available inventory for a product in a matching warehouse region.",
        ],
        method="POST",
        url_template="/inventory/{product_id}/reservations",
        auth_required=True,
        required_fields={
            "product_id": "string",
            "body_product_id": "string",
            "requested_quantity": "integer",
            "available_stock": "integer",
            "warehouse_region": "string",
            "customer_region": "string",
            "reservation_status": "enum:available|pending|cancelled|unavailable|locked",
            "allow_backorder": "boolean",
        },
        optional_fields={"reservation_note": "string"},
        constraints=[
            {
                "constraint_id": "inventory_quantity_positive",
                "constraint_text": "requested_quantity must be positive.",
                "constraint_variants": [
                    "requested_quantity must be positive.",
                    "Inventory reservations require a requested_quantity greater than zero.",
                    "The quantity being reserved must be at least one unit.",
                ],
                "constraint_type": "structural",
                "fields": ["requested_quantity"],
                "severity": "medium",
            },
            {
                "constraint_id": "inventory_stock_available",
                "constraint_text": "requested_quantity must not exceed available_stock unless allow_backorder is true.",
                "constraint_variants": [
                    "requested_quantity must not exceed available_stock unless allow_backorder is true.",
                    "The requested stock cannot be greater than available inventory unless backorder is allowed.",
                    "Reservations over available_stock require allow_backorder to be true.",
                ],
                "constraint_type": "semantic_domain",
                "fields": ["requested_quantity", "available_stock", "allow_backorder"],
                "severity": "high",
            },
            {
                "constraint_id": "inventory_region_match",
                "constraint_text": "warehouse_region must match customer_region for region-locked inventory.",
                "constraint_variants": [
                    "warehouse_region must match customer_region for region-locked inventory.",
                    "Region-locked stock can only be reserved for customers in the same region.",
                    "The warehouse and customer regions must align for this inventory reservation.",
                ],
                "constraint_type": "semantic_cross_field",
                "fields": ["warehouse_region", "customer_region"],
                "severity": "high",
            },
            {
                "constraint_id": "inventory_reservable_state",
                "constraint_text": "Inventory can only be reserved when reservation_status is available or pending.",
                "constraint_variants": [
                    "Inventory can only be reserved when reservation_status is available or pending.",
                    "Only available or pending inventory reservations may be updated.",
                    "Stock cannot be reserved when the reservation is cancelled, unavailable, or locked.",
                ],
                "constraint_type": "semantic_state",
                "fields": ["reservation_status"],
                "severity": "high",
            },
            {
                "constraint_id": "inventory_url_body_product_match",
                "constraint_text": "product_id in the URL must match body_product_id.",
                "constraint_variants": [
                    "product_id in the URL must match body_product_id.",
                    "The inventory URL and request body must refer to the same product.",
                    "A stock reservation cannot use different product IDs in the path and body.",
                ],
                "constraint_type": "semantic_cross_field",
                "fields": ["product_id", "body_product_id"],
                "severity": "high",
            },
        ],
    ),
    "payments_invoices": EndpointContract(
        endpoint_name="create_invoice_payment",
        description="Create a payment against an existing invoice.",
        description_variants=[
            "Create a payment against an existing invoice.",
            "Apply a customer payment to an invoice balance.",
            "Record an invoice payment with amount, currency, and method details.",
        ],
        method="POST",
        url_template="/invoices/{invoice_id}/payments",
        auth_required=True,
        required_fields={
            "invoice_id": "string",
            "body_invoice_id": "string",
            "payment_amount": "number",
            "invoice_total": "number",
            "amount_already_paid": "number",
            "invoice_status": "enum:open|partially_paid|cancelled|closed|refunded",
            "currency": "string",
            "expected_currency": "string",
            "payment_method": "enum:card|bank_transfer|wallet",
        },
        optional_fields={"payment_reference": "string"},
        constraints=[
            {
                "constraint_id": "payment_amount_positive",
                "constraint_text": "payment_amount must be positive.",
                "constraint_variants": [
                    "payment_amount must be positive.",
                    "Invoice payments require an amount greater than zero.",
                    "The submitted payment amount must be a positive value.",
                ],
                "constraint_type": "structural",
                "fields": ["payment_amount"],
                "severity": "medium",
            },
            {
                "constraint_id": "payment_not_over_invoice_total",
                "constraint_text": "payment_amount plus amount_already_paid must not exceed invoice_total.",
                "constraint_variants": [
                    "payment_amount plus amount_already_paid must not exceed invoice_total.",
                    "The new payment cannot push total paid above the invoice total.",
                    "Invoice payments may not overpay the remaining invoice balance.",
                ],
                "constraint_type": "semantic_domain",
                "fields": ["payment_amount", "amount_already_paid", "invoice_total"],
                "severity": "high",
            },
            {
                "constraint_id": "payment_currency_match",
                "constraint_text": "currency must match expected_currency.",
                "constraint_variants": [
                    "currency must match expected_currency.",
                    "The payment currency has to match the invoice currency.",
                    "Payments must use the expected currency for the invoice.",
                ],
                "constraint_type": "semantic_cross_field",
                "fields": ["currency", "expected_currency"],
                "severity": "high",
            },
            {
                "constraint_id": "payment_invoice_open_state",
                "constraint_text": "Payments can only be added when invoice_status is open or partially_paid.",
                "constraint_variants": [
                    "Payments can only be added when invoice_status is open or partially_paid.",
                    "Only open or partially paid invoices can accept new payments.",
                    "A cancelled, closed, or refunded invoice cannot receive another payment.",
                ],
                "constraint_type": "semantic_state",
                "fields": ["invoice_status"],
                "severity": "high",
            },
            {
                "constraint_id": "payment_method_allowed",
                "constraint_text": "payment_method must be one of card, bank_transfer, wallet.",
                "constraint_variants": [
                    "payment_method must be one of card, bank_transfer, wallet.",
                    "Invoice payments only support card, bank_transfer, or wallet methods.",
                    "Unsupported payment methods cannot be used for invoice payments.",
                ],
                "constraint_type": "structural",
                "fields": ["payment_method"],
                "severity": "medium",
            },
            {
                "constraint_id": "payment_url_body_invoice_match",
                "constraint_text": "invoice_id in the URL must match body_invoice_id.",
                "constraint_variants": [
                    "invoice_id in the URL must match body_invoice_id.",
                    "The invoice path and payment body must point to the same invoice.",
                    "A payment request cannot use different invoice IDs in the URL and body.",
                ],
                "constraint_type": "semantic_cross_field",
                "fields": ["invoice_id", "body_invoice_id"],
                "severity": "high",
            },
        ],
    ),
    "healthcare_appointments": EndpointContract(
        endpoint_name="schedule_patient_appointment",
        description="Schedule or update a healthcare appointment for a patient.",
        description_variants=[
            "Schedule or update a healthcare appointment for a patient.",
            "Request an appointment slot with a healthcare provider.",
            "Create a patient appointment using provider availability and coverage details.",
        ],
        method="POST",
        url_template="/patients/{patient_id}/appointments",
        auth_required=True,
        required_fields={
            "patient_id": "string",
            "body_patient_id": "string",
            "provider_id": "string",
            "appointment_type": "string",
            "requested_slot_start": "datetime",
            "requested_slot_end": "datetime",
            "provider_available_from": "datetime",
            "provider_available_until": "datetime",
            "patient_insurance_status": "enum:active|inactive|expired",
            "appointment_status": "enum:draft|requested|booked|cancelled|completed",
            "requires_referral": "boolean",
        },
        optional_fields={"referral_id": "string"},
        constraints=[
            {"constraint_id": "appt_patient_match", "constraint_text": "body_patient_id must match patient_id in the URL.", "constraint_variants": ["body_patient_id must match patient_id in the URL.", "The appointment body and patient URL must identify the same patient.", "A patient appointment cannot mix different path and body patient IDs."], "constraint_type": "semantic_cross_field", "fields": ["patient_id", "body_patient_id"], "severity": "high"},
            {"constraint_id": "appt_slot_order", "constraint_text": "requested_slot_start must be before requested_slot_end.", "constraint_variants": ["requested_slot_start must be before requested_slot_end.", "The requested appointment slot must start before it ends.", "Appointment start time must be earlier than appointment end time."], "constraint_type": "semantic_cross_field", "fields": ["requested_slot_start", "requested_slot_end"], "severity": "high"},
            {"constraint_id": "appt_provider_window", "constraint_text": "The requested slot must fall within provider availability.", "constraint_variants": ["The requested slot must fall within provider availability.", "Appointment times must stay inside the provider's available window.", "A provider cannot be booked outside provider_available_from and provider_available_until."], "constraint_type": "semantic_domain", "fields": ["requested_slot_start", "requested_slot_end", "provider_available_from", "provider_available_until"], "severity": "high"},
            {"constraint_id": "appt_referral_required", "constraint_text": "referral_id is required when requires_referral is true.", "constraint_variants": ["referral_id is required when requires_referral is true.", "Referral-based appointments need a referral_id.", "Appointments marked as requiring referral must include the referral identifier."], "constraint_type": "semantic_domain", "fields": ["requires_referral", "referral_id"], "severity": "high"},
            {"constraint_id": "appt_schedulable_state", "constraint_text": "Appointments can only be scheduled when appointment_status is draft or requested.", "constraint_variants": ["Appointments can only be scheduled when appointment_status is draft or requested.", "Only draft or requested appointments may be scheduled.", "Booked, cancelled, or completed appointments cannot be scheduled again."], "constraint_type": "semantic_state", "fields": ["appointment_status"], "severity": "high"},
            {"constraint_id": "appt_insurance_active", "constraint_text": "patient_insurance_status must be active for covered appointment types.", "constraint_variants": ["patient_insurance_status must be active for covered appointment types.", "Covered appointments require active patient insurance.", "Insurance-backed appointment types cannot be scheduled with inactive or expired insurance."], "constraint_type": "semantic_domain", "fields": ["appointment_type", "patient_insurance_status"], "severity": "high"},
        ],
    ),
    "course_registration": EndpointContract(
        endpoint_name="register_course_section",
        description="Register a student for a course section.",
        description_variants=[
            "Register a student for a course section.",
            "Submit a course registration request for a student.",
            "Enroll a student into a course section using eligibility and credit checks.",
        ],
        method="POST",
        url_template="/students/{student_id}/course-registrations",
        auth_required=True,
        required_fields={
            "student_id": "string",
            "body_student_id": "string",
            "course_id": "string",
            "section_id": "string",
            "completed_prerequisites": "list",
            "required_prerequisites": "list",
            "current_credits": "integer",
            "requested_course_credits": "integer",
            "max_allowed_credits": "integer",
            "enrollment_status": "enum:eligible|waitlisted|suspended|graduated|blocked",
            "registration_window_open": "boolean",
        },
        optional_fields={"advisor_override": "boolean"},
        constraints=[
            {"constraint_id": "course_student_match", "constraint_text": "body_student_id must match student_id in the URL.", "constraint_variants": ["body_student_id must match student_id in the URL.", "The registration body and student URL must identify the same student.", "Course registration cannot mix different path and body student IDs."], "constraint_type": "semantic_cross_field", "fields": ["student_id", "body_student_id"], "severity": "high"},
            {"constraint_id": "course_prerequisites", "constraint_text": "All required_prerequisites must be in completed_prerequisites.", "constraint_variants": ["All required_prerequisites must be in completed_prerequisites.", "Students must complete every required prerequisite before registration.", "A course section cannot be added when prerequisites are missing."], "constraint_type": "semantic_domain", "fields": ["completed_prerequisites", "required_prerequisites"], "severity": "high"},
            {"constraint_id": "course_credit_limit", "constraint_text": "current_credits plus requested_course_credits must not exceed max_allowed_credits.", "constraint_variants": ["current_credits plus requested_course_credits must not exceed max_allowed_credits.", "The requested course must not put the student over their credit limit.", "Total registered credits after this course must stay within max_allowed_credits."], "constraint_type": "semantic_cross_field", "fields": ["current_credits", "requested_course_credits", "max_allowed_credits"], "severity": "high"},
            {"constraint_id": "course_window_open", "constraint_text": "registration_window_open must be true.", "constraint_variants": ["registration_window_open must be true.", "Course registration is only allowed while the registration window is open.", "Closed registration windows block new course registrations."], "constraint_type": "semantic_state", "fields": ["registration_window_open"], "severity": "high"},
            {"constraint_id": "course_enrollment_state", "constraint_text": "enrollment_status must be eligible or waitlisted.", "constraint_variants": ["enrollment_status must be eligible or waitlisted.", "Only eligible or waitlisted students may register.", "Suspended, graduated, or blocked students cannot register for a new section."], "constraint_type": "semantic_state", "fields": ["enrollment_status"], "severity": "high"},
        ],
    ),
    "ticket_support_workflow": EndpointContract(
        endpoint_name="transition_support_ticket",
        description="Move a support ticket to a new workflow status.",
        description_variants=[
            "Move a support ticket to a new workflow status.",
            "Transition a customer support ticket through the service workflow.",
            "Update ticket workflow state with agent, priority, and resolution context.",
        ],
        method="PATCH",
        url_template="/tickets/{ticket_id}/workflow",
        auth_required=True,
        required_fields={
            "ticket_id": "string",
            "body_ticket_id": "string",
            "current_status": "enum:new|triaged|in_progress|escalated|resolved|closed",
            "target_status": "enum:new|triaged|in_progress|escalated|resolved|closed",
            "assigned_agent_id": "string",
            "acting_user_id": "string",
            "customer_visible": "boolean",
            "priority": "enum:low|normal|high",
            "escalation_allowed": "boolean",
        },
        optional_fields={"resolution_code": "string"},
        constraints=[
            {"constraint_id": "ticket_id_match", "constraint_text": "body_ticket_id must match ticket_id in the URL.", "constraint_variants": ["body_ticket_id must match ticket_id in the URL.", "The workflow body and ticket path must refer to the same ticket.", "Ticket workflow transitions cannot mix path and body ticket IDs."], "constraint_type": "semantic_cross_field", "fields": ["ticket_id", "body_ticket_id"], "severity": "high"},
            {"constraint_id": "ticket_status_change", "constraint_text": "target_status must be different from current_status.", "constraint_variants": ["target_status must be different from current_status.", "A workflow transition must actually change the ticket status.", "The target ticket status cannot be the same as the current status."], "constraint_type": "semantic_cross_field", "fields": ["current_status", "target_status"], "severity": "medium"},
            {"constraint_id": "ticket_resolution_code", "constraint_text": "Tickets can move to resolved only when resolution_code is provided.", "constraint_variants": ["Tickets can move to resolved only when resolution_code is provided.", "Resolved tickets require a resolution code.", "A ticket cannot be marked resolved without resolution context."], "constraint_type": "semantic_domain", "fields": ["target_status", "resolution_code"], "severity": "high"},
            {"constraint_id": "ticket_assigned_agent", "constraint_text": "Only assigned agents may move a ticket to in_progress or resolved.", "constraint_variants": ["Only assigned agents may move a ticket to in_progress or resolved.", "Agent-owned workflow moves require acting_user_id to match assigned_agent_id.", "Unassigned users cannot progress or resolve a ticket."], "constraint_type": "semantic_domain", "fields": ["assigned_agent_id", "acting_user_id", "target_status"], "severity": "high"},
            {"constraint_id": "ticket_escalation_rule", "constraint_text": "High-priority tickets require escalation_allowed true before moving to escalated.", "constraint_variants": ["High-priority tickets require escalation_allowed true before moving to escalated.", "Escalating a high-priority ticket requires escalation permission.", "High-priority escalation is blocked unless escalation_allowed is true."], "constraint_type": "semantic_domain", "fields": ["priority", "target_status", "escalation_allowed"], "severity": "high"},
            {"constraint_id": "ticket_closed_transition", "constraint_text": "Closed tickets cannot transition back to in_progress.", "constraint_variants": ["Closed tickets cannot transition back to in_progress.", "A closed ticket cannot be reopened directly into in_progress.", "The workflow blocks moving closed tickets back to active work."], "constraint_type": "semantic_state", "fields": ["current_status", "target_status"], "severity": "high"},
        ],
    ),
    "shipping_returns": EndpointContract(
        endpoint_name="create_shipping_return",
        description="Create a shipping return authorization for a delivered shipment.",
        description_variants=[
            "Create a shipping return authorization for a delivered shipment.",
            "Request a logistics return for shipped goods.",
            "Authorize a return shipment using delivery state, return window, and label rules.",
        ],
        method="POST",
        url_template="/shipments/{shipment_id}/returns",
        auth_required=True,
        required_fields={
            "shipment_id": "string",
            "body_shipment_id": "string",
            "delivery_status": "enum:delivered|in_transit|lost|returned|cancelled",
            "days_since_delivery": "integer",
            "return_window_days": "integer",
            "item_condition": "enum:new|opened|damaged|defective",
            "return_reason": "string",
            "prepaid_label_requested": "boolean",
            "destination_country": "string",
            "return_country": "string",
        },
        optional_fields={"carrier_code": "string"},
        constraints=[
            {"constraint_id": "return_shipment_match", "constraint_text": "body_shipment_id must match shipment_id in the URL.", "constraint_variants": ["body_shipment_id must match shipment_id in the URL.", "The return body and shipment URL must refer to the same shipment.", "A return authorization cannot mix different path and body shipment IDs."], "constraint_type": "semantic_cross_field", "fields": ["shipment_id", "body_shipment_id"], "severity": "high"},
            {"constraint_id": "return_delivered_state", "constraint_text": "Returns are only allowed when delivery_status is delivered.", "constraint_variants": ["Returns are only allowed when delivery_status is delivered.", "Only delivered shipments can enter the return flow.", "Shipments still in transit, lost, returned, or cancelled cannot start a new return."], "constraint_type": "semantic_state", "fields": ["delivery_status"], "severity": "high"},
            {"constraint_id": "return_window", "constraint_text": "days_since_delivery must not exceed return_window_days.", "constraint_variants": ["days_since_delivery must not exceed return_window_days.", "Returns must be requested within the allowed return window.", "A shipment is not returnable after the return window has expired."], "constraint_type": "semantic_domain", "fields": ["days_since_delivery", "return_window_days"], "severity": "high"},
            {"constraint_id": "return_condition_reason", "constraint_text": "Damaged items require return_reason to be damaged or defective.", "constraint_variants": ["Damaged items require return_reason to be damaged or defective.", "A damaged item return must use a damage-related reason.", "The return reason must match damaged or defective item conditions."], "constraint_type": "semantic_domain", "fields": ["item_condition", "return_reason"], "severity": "high"},
            {"constraint_id": "return_label_country", "constraint_text": "Prepaid labels are only available when destination_country matches return_country.", "constraint_variants": ["Prepaid labels are only available when destination_country matches return_country.", "A prepaid return label requires destination and return countries to match.", "Cross-country returns cannot request a prepaid label through this endpoint."], "constraint_type": "semantic_cross_field", "fields": ["prepaid_label_requested", "destination_country", "return_country"], "severity": "high"},
        ],
    ),
    "subscription_plan_changes": EndpointContract(
        endpoint_name="change_subscription_plan",
        description="Change a customer subscription plan.",
        description_variants=[
            "Change a customer subscription plan.",
            "Schedule a subscription plan transition for an active account.",
            "Update a subscription using billing status, invoice, and effective-date rules.",
        ],
        method="PATCH",
        url_template="/subscriptions/{subscription_id}/plan",
        auth_required=True,
        required_fields={
            "subscription_id": "string",
            "body_subscription_id": "string",
            "current_plan": "enum:basic|pro|enterprise",
            "target_plan": "enum:basic|pro|enterprise",
            "billing_status": "enum:valid|past_due|failed",
            "account_status": "enum:active|paused|cancelled",
            "requested_effective_date": "date",
            "current_billing_period_end": "date",
            "has_unpaid_invoice": "boolean",
            "downgrade_allowed": "boolean",
        },
        optional_fields={"change_reason": "string"},
        constraints=[
            {"constraint_id": "subscription_id_match", "constraint_text": "body_subscription_id must match subscription_id in the URL.", "constraint_variants": ["body_subscription_id must match subscription_id in the URL.", "The plan-change body and URL must refer to the same subscription.", "Subscription plan changes cannot mix different path and body subscription IDs."], "constraint_type": "semantic_cross_field", "fields": ["subscription_id", "body_subscription_id"], "severity": "high"},
            {"constraint_id": "subscription_plan_changes_only", "constraint_text": "target_plan must be different from current_plan.", "constraint_variants": ["target_plan must be different from current_plan.", "A plan-change request must choose a different target plan.", "The requested plan cannot be the same as the active plan."], "constraint_type": "semantic_cross_field", "fields": ["current_plan", "target_plan"], "severity": "medium"},
            {"constraint_id": "subscription_account_active", "constraint_text": "Plan changes are allowed only when account_status is active.", "constraint_variants": ["Plan changes are allowed only when account_status is active.", "Only active subscription accounts can change plans.", "Paused or cancelled accounts cannot process plan transitions."], "constraint_type": "semantic_state", "fields": ["account_status"], "severity": "high"},
            {"constraint_id": "subscription_billing_valid", "constraint_text": "Plan upgrades require billing_status to be valid.", "constraint_variants": ["Plan upgrades require billing_status to be valid.", "A subscription cannot upgrade while billing is past_due or failed.", "Upgrade requests require a valid billing state."], "constraint_type": "semantic_state", "fields": ["current_plan", "target_plan", "billing_status"], "severity": "high"},
            {"constraint_id": "subscription_downgrade_timing", "constraint_text": "Downgrades before current_billing_period_end require downgrade_allowed true.", "constraint_variants": ["Downgrades before current_billing_period_end require downgrade_allowed true.", "Early downgrades need downgrade_allowed to be true.", "A plan downgrade before the billing period ends is blocked without downgrade permission."], "constraint_type": "semantic_domain", "fields": ["requested_effective_date", "current_billing_period_end", "downgrade_allowed"], "severity": "high"},
            {"constraint_id": "subscription_unpaid_invoice", "constraint_text": "Plan changes are blocked when has_unpaid_invoice is true.", "constraint_variants": ["Plan changes are blocked when has_unpaid_invoice is true.", "Open unpaid invoices prevent subscription plan changes.", "A customer must clear unpaid invoices before changing plans."], "constraint_type": "semantic_domain", "fields": ["has_unpaid_invoice"], "severity": "high"},
        ],
    ),
}


def get_contract(endpoint_family: str) -> EndpointContract:
    return CONTRACTS[endpoint_family]
