# Stage 2 Seen Families

Stage 2 expanded the seen endpoint families from 3 to 5.

Implemented seen families:

- `refunds_orders`
- `booking_reservation`
- `user_permissions`
- `inventory_product_search`
- `payments_invoices`

Quality improvements:

- balanced endpoint-family sampling
- stronger label distribution controls
- more natural-language paraphrases
- duplicate and near-duplicate reporting
- manual review exports
- shortcut-view generation
- malformed-body diversity
- validator boundary fixes for missing fields and cross-field labels

Stage 2.5 added a quality gate around duplicates, shortcuts, and manual-review readiness.
