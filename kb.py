"""Task 2 - Nykaa knowledge base (12 documents, 4 sentences each, fictional policies
written for this exercise)."""

DOCUMENTS = [
 {"doc_id": "return_window", "title": "Return window by category",
  "text": "Nykaa accepts returns within a window that depends on the product category. Beauty products can be returned within 15 days of delivery as long as they are unopened and the seal is intact. Apparel and Footwear can be returned within 15 days if tags are attached and the item is unworn. Electronics have a 7-day return window, and Home items have a 10-day return window."},
 {"doc_id": "cod_refund", "title": "COD refund timelines",
  "text": "Refunds for Cash on Delivery orders are paid by bank transfer, because there is no card or wallet payment to reverse. The customer must share valid bank account details after the returned item is picked up. The refund is processed within 5 to 7 working days after the item passes quality check. Customers who prefer faster money can choose Nykaa wallet credit, which is added within 24 hours."},
 {"doc_id": "delivery_sla", "title": "Delivery SLAs",
  "text": "Standard delivery in metro cities takes 2 to 4 days, while other locations take 5 to 7 days. Express delivery is available for selected pin codes and arrives within 24 hours. An order that has shipped but is not delivered within 3 days after the promised date is treated as a delayed shipment. Weather disruptions and strikes extend the promised date."},
 {"doc_id": "reverse_pickup", "title": "Reverse-pickup eligibility",
  "text": "Reverse pickup is available when the delivery pin code is serviceable by our courier partners. An item is eligible if it is still inside its category return window and is not a final-sale product. The courier makes up to two pickup attempts on consecutive days. If the pin code is not serviceable, the customer can use a drop-off courier point and the shipping cost is reimbursed."},
 {"doc_id": "warranty", "title": "Warranty terms by category",
  "text": "Electronics carry a 12-month manufacturer warranty that starts from the delivery date. Home products carry a 6-month warranty against manufacturing defects. Apparel, Footwear and Beauty products have no warranty, but manufacturing defects can be raised through the damaged-item claim process. Warranty does not cover physical damage, water damage or misuse."},
 {"doc_id": "order_cancellation", "title": "Order-cancellation policy",
  "text": "An order in Placed status can be cancelled free of charge until it is packed. Once an order is Shipped it can no longer be cancelled, so the customer should refuse delivery or request a return after receiving it. Prepaid cancellations are refunded to the original payment method within 4 working days. Cancelled Cash on Delivery orders need no refund because nothing was paid."},
 {"doc_id": "loyalty_points", "title": "Loyalty-points redemption",
  "text": "Customers earn 1 loyalty point for every 100 rupees spent on delivered orders. Each point is worth 1 rupee, and at least 200 points are needed before any redemption. Points can cover up to 20 percent of an order's value and expire 12 months after they are earned. Points used on a returned order are restored to the account within 48 hours of the refund."},
 {"doc_id": "payment_failure", "title": "Payment failure and retry policy",
  "text": "If a payment fails but the money is debited, the amount is reversed automatically within 3 to 5 working days. A customer can retry payment up to three times on the same order within 30 minutes. After that the order is auto-cancelled and the cart is restored. Customers should avoid making a duplicate payment while a transaction shows as pending."},
 {"doc_id": "size_exchange", "title": "Size-exchange policy",
  "text": "Size exchanges are available for Apparel and Footwear within 15 days of delivery, subject to stock. The first exchange on an order is free, and later exchanges carry a ₹99 reverse shipping fee. If the requested size is out of stock, the customer receives a refund instead. Items must be unworn with all tags attached."},
 {"doc_id": "damaged_item", "title": "Damaged-item claim process",
  "text": "If an item arrives damaged or defective, the customer must raise a claim within 48 hours of delivery. The claim needs a photo or unboxing video showing the damage along with the shipping label. After verification, Nykaa sends a free replacement or a full refund, whichever the customer prefers. Claims raised after 48 hours go to manual review."},
 {"doc_id": "international_shipping", "title": "International shipping restrictions",
  "text": "International shipping is available only to selected countries. Electronics and flammable beauty products such as aerosols cannot be shipped abroad. International orders cannot be cancelled once shipped and are not eligible for reverse pickup. Customs duties and import taxes are paid by the recipient."},
 {"doc_id": "escalation_matrix", "title": "Customer-support escalation matrix",
  "text": "Level 1 is the support agent, who owns every query for the first 24 hours. If a query is unresolved after 24 hours or the order is flagged as delayed, it moves to Level 2, the senior support team, with a 48-hour target. Payment disputes and claims above ₹10,000 go straight to Level 3, the support manager. Legal threats and safety complaints are sent to the grievance officer immediately."},
]
