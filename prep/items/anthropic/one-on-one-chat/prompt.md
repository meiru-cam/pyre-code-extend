Design the backend for direct messaging between exactly two people: no group conversations and no channels. A *conversation* is the entire exchange between one specific pair of users; there is exactly one conversation per pair, and it comes into being with the first message either side sends the other. Each user is connected from at most one device at a time. A message's body is plain text of up to 4 KiB.

While both participants are connected, a message must reach the recipient's screen with low latency. If the recipient is not connected when a message is sent, the system must hold onto the message and deliver it the next time the recipient connects, without the recipient having to know to ask for it. Within one conversation, both participants must see its messages in a single order — the order in which the server accepted them, from whichever side sent them — even when a client's connection happens to deliver them in a different order and has to reconcile before displaying them. A message the sender successfully submitted must never be stored twice, even if the client retries the request after not hearing back, and a message already shown to a recipient must never be shown to them again, even if the transport underneath redelivers it.

The sender is told once a message has been delivered — a *delivery receipt* — meaning the recipient's client has durably received it, independent of whether the recipient has looked at it. Whether the recipient has looked at a message — a *read receipt* — is out of scope. Each user can also see, for the person they are messaging, whether that person is currently connected; this *presence* signal is allowed to lag an actual disconnect by up to the detection bound your design states, and it is a hint for the user interface, not a guarantee checked anywhere else in the system.

Scale this design for:

- 250,000,000 registered users, of whom 50,000,000 are active on a typical day.
- An active user sends an average of 24 messages a day.
- Peak traffic runs at about 3 times the daily average, spread across evening hours in every time zone rather than concentrated the way a workplace tool's business-hours peak is.
- At peak, 8% of daily active users are connected at once.
- Target end-to-end delivery latency while both users are connected: 95th percentile under 300 ms.
- Target availability: 99.9%.
- Conversation history is retained indefinitely.

In scope: one-to-one messaging between exactly two users; real-time delivery while both are connected; durable delivery to a recipient who was offline when a message was sent; a single order for each conversation's messages; deduplication, so that neither a retried send nor a redelivered push ever produces a duplicate; delivery receipts; online presence; and horizontal scaling of the tier holding client connections. Out of scope: group conversations and channels; more than one connected device per user; searching message history; attachments of any kind (images, files, voice notes); and end-to-end encryption — mention it only as a follow-up.

Produce:

- A requirements and scale estimate: the average and peak message rate, the number of concurrent connections and the fleet of instances holding them that requires (state your assumption for how many connections one instance holds), and the annual growth of message storage (state your assumption for the size of a stored message).
- A data model — for conversations, messages, and however you choose to hold a message that has not yet reached its recipient — and the API: the frames exchanged over a client's persistent connection, and the REST endpoints used to send without an open connection, to page through a conversation's history, and to catch up after being away.
- An architecture diagram, and a walk-through of one message from the sender's client to the recipient's, covering both the case where the recipient is already connected and the case where they are not.
- Deep dives into: (a) the tier that holds client connections — how a client connects, and how the system finds the connection belonging to a given recipient once there is more than one instance holding connections; (b) delivery semantics — the delivery guarantee your design actually provides, how it is acknowledged, and what stops a duplicate from ever reaching a user; (c) the queue that holds a message for a recipient who is not connected, and what happens when they reconnect; (d) presence — how a disconnect is detected and how that information reaches the people who need it; (e) the delivery bus that carries a message from the sender's side of the system to the recipient's — compare Kafka and Redis for this role specifically, say which fits this design and why, and state when the other would be the better choice.
