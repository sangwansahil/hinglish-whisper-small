const list = document.querySelector("#corrections");

async function load() {
  const res = await fetch("/corrections");
  const rows = await res.json();
  if (!rows.length) {
    list.innerHTML = `<p class="empty">No corrections saved yet.</p>`;
    return;
  }
  list.innerHTML = "";
  for (const row of rows) {
    const card = document.createElement("article");
    card.className = "correction";
    card.innerHTML = `
      <textarea>${row.text}</textarea>
      <div class="correction-actions">
        <span>${row.id}</span>
        <button type="button" data-action="save">Save</button>
        <button type="button" data-action="delete">Delete</button>
      </div>
    `;
    card.querySelector('[data-action="save"]').onclick = async () => {
      const text = card.querySelector("textarea").value;
      await fetch(`/corrections/${row.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      load();
    };
    card.querySelector('[data-action="delete"]').onclick = async () => {
      await fetch(`/corrections/${row.id}`, { method: "DELETE" });
      load();
    };
    list.append(card);
  }
}

load();
