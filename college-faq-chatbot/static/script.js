/**
 * =============================================================================
 * CampusExam AI - Interactive Chat & NLP Inspection Logic
 * =============================================================================
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const chatMessages = document.getElementById('chatMessages');
  const chatForm = document.getElementById('chatForm');
  const userInput = document.getElementById('userInput');
  const btnSend = document.getElementById('btnSend');
  const typingIndicator = document.getElementById('typingIndicator');
  const btnClearChat = document.getElementById('btnClearChat');
  const charCount = document.getElementById('charCount');

  // Modals
  const btnHowItWorks = document.getElementById('btnHowItWorks');
  const btnFaqDirectory = document.getElementById('btnFaqDirectory');
  const btnVivaGuide = document.getElementById('btnVivaGuide');

  const modalHowItWorks = document.getElementById('modalHowItWorks');
  const modalFaqDirectory = document.getElementById('modalFaqDirectory');
  const modalVivaGuide = document.getElementById('modalVivaGuide');

  const faqAccordionList = document.getElementById('faqAccordionList');
  const faqSearchInput = document.getElementById('faqSearchInput');

  let allFaqIntents = [];

  // ===========================================================================
  // UTILITY: Get formatted time
  // ===========================================================================
  function getCurrentTime() {
    const now = new Date();
    return now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }

  // ===========================================================================
  // AUTO-SCROLL TO BOTTOM
  // ===========================================================================
  function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  // ===========================================================================
  // AUTO-RESIZE TEXTAREA & CHAR COUNTER
  // ===========================================================================
  userInput.addEventListener('input', () => {
    userInput.style.height = 'auto';
    userInput.style.height = Math.min(userInput.scrollHeight, 120) + 'px';
    const len = userInput.value.length;
    charCount.textContent = `${len} / 400`;
  });

  // Handle Enter to submit (Shift+Enter for new line)
  userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      chatForm.dispatchEvent(new Event('submit'));
    }
  });

  // ===========================================================================
  // RENDER USER MESSAGE
  // ===========================================================================
  function appendUserMessage(text) {
    const row = document.createElement('div');
    row.className = 'message-row user-row';
    
    const time = getCurrentTime();

    row.innerHTML = `
      <div class="avatar user-avatar">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
          <circle cx="12" cy="7" r="4"></circle>
        </svg>
      </div>
      <div class="message-content-wrapper">
        <div class="bubble user-bubble">${escapeHtml(text)}</div>
        <span class="message-time">${time}</span>
      </div>
    `;

    chatMessages.appendChild(row);
    scrollToBottom();
  }

  // ===========================================================================
  // RENDER BOT MESSAGE
  // ===========================================================================
  function appendBotMessage(data) {
    const row = document.createElement('div');
    row.className = 'message-row bot-row';

    const time = getCurrentTime();
    const isFallback = data.status === 'fallback' || data.confidence < 0.22;
    
    // Confidence Pill Class
    let confClass = 'conf-high';
    if (data.confidence < 0.35) {
      confClass = 'conf-low';
    } else if (data.confidence < 0.65) {
      confClass = 'conf-mid';
    }

    const intentTag = data.intent ? `#${data.intent}` : '#general';
    const confPercent = data.confidence_percentage || `${Math.round((data.confidence || 0) * 100)}%`;

    // Suggestions / Follow-up Quick Chips
    let chipsHtml = '';
    if (data.quick_chips && data.quick_chips.length > 0) {
      chipsHtml = `
        <div class="quick-chips-grid" style="margin-top: 0.6rem;">
          ${data.quick_chips.map(chip => `<button class="chip-btn" data-query="${escapeHtml(chip)}">${escapeHtml(chip)}</button>`).join('')}
        </div>
      `;
    }

    // NLP Inspector details
    let nlpDrawerHtml = '';
    if (data.nlp_breakdown) {
      const b = data.nlp_breakdown;
      nlpDrawerHtml = `
        <div class="nlp-inspect-drawer">
          <div class="nlp-detail-row"><span class="nlp-label">Cleaned Query:</span><span class="nlp-value">${escapeHtml(b.cleaned_query || '-')}</span></div>
          <div class="nlp-detail-row"><span class="nlp-label">Filtered Tokens:</span><span class="nlp-value">[${(b.processed_tokens || []).join(', ')}]</span></div>
          <div class="nlp-detail-row"><span class="nlp-label">Matched Terms:</span><span class="nlp-value">[${(b.matched_vocabulary_terms || []).join(', ')}]</span></div>
          <div class="nlp-detail-row"><span class="nlp-label">Cosine Similarity:</span><span class="nlp-value">${b.similarity_score}</span></div>
          <div class="nlp-detail-row"><span class="nlp-label">Intent Category:</span><span class="nlp-value">${b.category || '-'}</span></div>
        </div>
      `;
    }

    row.innerHTML = `
      <div class="avatar bot-avatar">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M12 2a2 2 0 0 1 2 2v2a2 2 0 0 1-2 2 2 2 0 0 1-2-2V4a2 2 0 0 1 2-2z"/>
          <rect x="4" y="8" width="16" height="12" rx="2"/>
          <circle cx="9" cy="13" r="1.5" fill="currentColor"/>
          <circle cx="15" cy="13" r="1.5" fill="currentColor"/>
          <path d="M9 17h6"/>
        </svg>
      </div>
      <div class="message-content-wrapper">
        <div class="bubble bot-bubble">
          <p class="bot-answer-text">${formatAnswerText(data.answer)}</p>
          
          ${chipsHtml}

          <div class="bot-meta-bar">
            <div class="meta-tags">
              <span class="intent-pill">${escapeHtml(intentTag)}</span>
              <span class="confidence-pill ${confClass}">Score: ${confPercent}</span>
            </div>
            ${data.nlp_breakdown ? `
              <button class="btn-inspect-nlp">
                <span>Inspect NLP</span>
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"></polyline></svg>
              </button>
            ` : ''}
          </div>

          ${nlpDrawerHtml}
        </div>
        <span class="message-time">${time}</span>
      </div>
    `;

    // Attach toggle listener to Inspect NLP button
    const inspectBtn = row.querySelector('.btn-inspect-nlp');
    const nlpDrawer = row.querySelector('.nlp-inspect-drawer');
    if (inspectBtn && nlpDrawer) {
      inspectBtn.addEventListener('click', () => {
        const isShown = nlpDrawer.classList.toggle('active');
        inspectBtn.querySelector('svg').style.transform = isShown ? 'rotate(180deg)' : 'rotate(0deg)';
        scrollToBottom();
      });
    }

    chatMessages.appendChild(row);
    scrollToBottom();
  }

  // ===========================================================================
  // SUBMIT QUESTION VIA AJAX TO /chat
  // ===========================================================================
  async function handleSendMessage(query) {
    const message = (query || userInput.value).trim();
    if (!message) return;

    // Reset input
    userInput.value = '';
    userInput.style.height = 'auto';
    charCount.textContent = '0 / 400';
    btnSend.disabled = true;

    // Display user message
    appendUserMessage(message);

    // Show typing indicator
    typingIndicator.style.display = 'flex';
    scrollToBottom();

    try {
      const response = await fetch('/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ message: message })
      });

      if (!response.ok) {
        throw new Error(`Server returned HTTP ${response.status}`);
      }

      const data = await response.json();

      // Brief delay to make the AI interaction feel natural
      setTimeout(() => {
        typingIndicator.style.display = 'none';
        appendBotMessage(data);
        btnSend.disabled = false;
        userInput.focus();
      }, 350);

    } catch (err) {
      console.error('Chat error:', err);
      setTimeout(() => {
        typingIndicator.style.display = 'none';
        appendBotMessage({
          status: 'error',
          intent: 'network_error',
          answer: 'Unable to communicate with the CampusExam AI backend. Please make sure the Flask server is running on http://127.0.0.1:5000.',
          confidence: 0.0,
          confidence_percentage: '0%',
          quick_chips: ['Exam Dates', 'Hall Ticket', 'Results']
        });
        btnSend.disabled = false;
        userInput.focus();
      }, 350);
    }
  }

  // Form Submit Listener
  chatForm.addEventListener('submit', (e) => {
    e.preventDefault();
    handleSendMessage();
  });

  // Global Chip and Button Click Listener (Event Delegation)
  document.addEventListener('click', (e) => {
    const target = e.target.closest('.chip-btn, .strip-btn, .btn-ask-this');
    if (target) {
      const query = target.getAttribute('data-query');
      if (query) {
        // If modal was open, close it
        closeAllModals();
        handleSendMessage(query);
      }
    }
  });

  // ===========================================================================
  // CLEAR CONVERSATION
  // ===========================================================================
  btnClearChat.addEventListener('click', () => {
    const keepWelcome = chatMessages.querySelector('.welcome-message-row');
    chatMessages.innerHTML = '';
    if (keepWelcome) {
      chatMessages.appendChild(keepWelcome);
    }
    userInput.focus();
  });

  // ===========================================================================
  // MODAL CONTROLS
  // ===========================================================================
  function openModal(modal) {
    modal.classList.add('active');
  }

  function closeAllModals() {
    [modalHowItWorks, modalFaqDirectory, modalVivaGuide].forEach(m => m.classList.remove('active'));
  }

  btnHowItWorks.addEventListener('click', () => openModal(modalHowItWorks));
  btnFaqDirectory.addEventListener('click', () => {
    openModal(modalFaqDirectory);
    loadFaqDirectory();
  });
  btnVivaGuide.addEventListener('click', () => openModal(modalVivaGuide));

  // Modal Close Buttons and Backdrop Click
  document.querySelectorAll('[data-close]').forEach(btn => {
    btn.addEventListener('click', () => {
      const modalId = btn.getAttribute('data-close');
      const targetModal = document.getElementById(modalId);
      if (targetModal) targetModal.classList.remove('active');
    });
  });

  [modalHowItWorks, modalFaqDirectory, modalVivaGuide].forEach(modal => {
    modal.addEventListener('click', (e) => {
      if (e.target === modal) {
        modal.classList.remove('active');
      }
    });
  });

  // ===========================================================================
  // FAQ DIRECTORY LOADER & SEARCH
  // ===========================================================================
  async function loadFaqDirectory() {
    if (allFaqIntents.length > 0) return; // already loaded

    try {
      const res = await fetch('/api/faq');
      const data = await res.json();
      allFaqIntents = data.intents || [];
      renderFaqAccordion(allFaqIntents);
    } catch (err) {
      faqAccordionList.innerHTML = `<p style="color: #dc2626; padding: 1rem;">Failed to load FAQ topics: ${err.message}</p>`;
    }
  }

  function renderFaqAccordion(intents) {
    if (!intents || intents.length === 0) {
      faqAccordionList.innerHTML = `<p style="padding: 1rem; color: #64748b;">No FAQ topics found.</p>`;
      return;
    }

    faqAccordionList.innerHTML = intents.map((item, index) => {
      const sampleQ = (item.patterns && item.patterns[0]) || item.intent;
      return `
        <div class="faq-item">
          <button class="faq-question-btn" data-toggle="faq-${index}">
            <span><strong>[${escapeHtml(item.category || 'General')}]</strong> ${escapeHtml(capitalizeFirstLetter(sampleQ))}</span>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"></polyline></svg>
          </button>
          <div class="faq-answer-pane" id="faq-${index}">
            <p>${formatAnswerText(item.answer)}</p>
            <div style="margin-top: 0.6rem; display: flex; gap: 0.5rem; flex-wrap: wrap; align-items: center;">
              <span style="font-size: 0.72rem; color: #64748b;">Try asking:</span>
              <button class="btn-ask-this" data-query="${escapeHtml(sampleQ)}">Ask: "${escapeHtml(sampleQ)}"</button>
            </div>
          </div>
        </div>
      `;
    }).join('');

    // Attach toggle listeners
    faqAccordionList.querySelectorAll('.faq-question-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const paneId = btn.getAttribute('data-toggle');
        const pane = document.getElementById(paneId);
        const isOpen = pane.classList.toggle('active');
        btn.querySelector('svg').style.transform = isOpen ? 'rotate(180deg)' : 'rotate(0deg)';
      });
    });
  }

  // FAQ Live Search Filter
  faqSearchInput.addEventListener('input', () => {
    const q = faqSearchInput.value.toLowerCase().trim();
    if (!q) {
      renderFaqAccordion(allFaqIntents);
      return;
    }

    const filtered = allFaqIntents.filter(item => {
      const inCategory = (item.category || '').toLowerCase().includes(q);
      const inAnswer = (item.answer || '').toLowerCase().includes(q);
      const inPatterns = (item.patterns || []).some(p => p.toLowerCase().includes(q));
      return inCategory || inAnswer || inPatterns;
    });

    renderFaqAccordion(filtered);
  });

  // ===========================================================================
  // HELPERS
  // ===========================================================================
  function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  function formatAnswerText(text) {
    if (!text) return '';
    // Bold numbered points like (1), (2)
    return escapeHtml(text)
      .replace(/\(([1-9])\)/g, '<br><strong>($1)</strong>');
  }

  function capitalizeFirstLetter(string) {
    if (!string) return '';
    return string.charAt(0).toUpperCase() + string.slice(1);
  }
});
