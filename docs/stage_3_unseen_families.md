# Stage 3 Unseen Families

Stage 3 implemented the held-out unseen endpoint families for `unseen_family_test`.

Implemented unseen families:

- `healthcare_appointments`
- `course_registration`
- `ticket_support_workflow`
- `shipping_returns`
- `subscription_plan_changes`

The unseen families were designed to avoid being simple renamed copies of seen families. They use different domains, field names, request structures, status values, and semantic reasoning patterns.

Stage 3.5 added leakage and shortcut checks:

- seen-vs-unseen overlap report
- cross-split duplicate and near-duplicate checks
- unseen shortcut views
- unseen manual-review exports
- clarified `invalid_value_range` documentation for enum-like invalid values
