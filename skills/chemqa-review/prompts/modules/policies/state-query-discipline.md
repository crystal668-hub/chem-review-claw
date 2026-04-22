State-query discipline:

- Prefer the compact cached snapshot helper over raw `status` / `next-action` JSON when the runtime wrapper asks you to reason about current state.
- The runtime wrapper owns ordinary waiting, sleeping, polling, and `advance` calls.
- Query state only when the runtime wrapper asks you for a concrete artifact-generation or diagnosis step.
- After the runtime wrapper performs a state-changing transport command (`submit-proposal`, `submit-review`, `submit-rebuttal`, or `advance`), it will verify the new state itself. Do not repeat transport confirmation work inside the model.
- Do not poll `status` and `next-action` back-to-back inside the model unless the runtime wrapper explicitly asks for deeper diagnosis.
- If the last wrapper-provided snapshot says the state is unchanged, continue the requested artifact task instead of inventing your own polling loop.
- Avoid rapid polling loops entirely. In ChemQA runs, waiting is infrastructure behavior, not model work.
