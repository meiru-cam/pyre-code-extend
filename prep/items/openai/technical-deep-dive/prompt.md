This round is 60 minutes. Before it, you prepare a short slide deck about one technical project you drove yourself and are ready to defend against detailed pushback: the architecture, the key technical decisions and their trade-offs, and the measurable impact. You present the deck, and the interviewer interrupts throughout with follow-up questions that push past the slides into implementation detail.

Expect the follow-ups to fall into five groups.

### Architecture and key decisions

- Before the architecture: what problem were you solving, and what did you decide would count as success before you started building it?
- Walk through the architecture: the main components and how a request or a piece of data moves through them.
- For each key decision, what alternatives did you consider, and why did you rule them out?
- For each key decision, which of these did you trade off against which: accuracy, latency, cost, scalability, reliability, complexity, or how fast you could iterate?
- Why did you use your actual choice instead of a specific named alternative, for a specific piece of the system?
- Which part of the design would you defend most strongly if someone pushed back on it?
- Where did you compromise on what you considered the ideal design because of a real constraint — time, team size, or an existing system you had to integrate with?

### Scale and limits

- What happens to the system if traffic or data volume grows 10x? Which component breaks first?
- What was the actual bottleneck once the system was in production, and how did you find it?
- If you had to support 100x instead of 10x, would the same architecture still work, or would it need to change shape?
- What is the failure mode when a specific resource — memory, a queue, a rate limit — is exhausted, and how does the system behave then?

### Measuring impact

- What metric did you use to show the project worked, and why that metric rather than another one?
- What was the number before and after, and what did you compare it against — a previous baseline, a control group, a rollback?
- How much of the result is your individual contribution, and how much is the team's?
- How do you know the change in the metric was caused by your project and not by something else that changed at the same time?

### Retrospective

- If you started this project again today, what would you do differently?
- What was the single biggest mistake in the project, and when did you realize it?
- What did you learn that you have since applied to a later project?
- Was there a simpler design that would have captured most of the benefit for a fraction of the effort?

### Where AI could help

- Which part of the system, if any, could be replaced or improved with a machine-learned model?
- If the project itself had nothing to do with AI, where in the pipeline could a model reduce manual or heuristic work today?
- Where would the training and evaluation data for that model come from?
- What is the risk of adding a model there, and how would you bound it?
