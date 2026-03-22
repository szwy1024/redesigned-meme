<script setup lang="ts">
import { ref, computed } from 'vue'
import { analyzeText, type AnalyzeResponse } from './api/sentiment'

// ============== State ==============
const inputText = ref('')
const isLoading = ref(false)
const errorMessage = ref('')
const result = ref<AnalyzeResponse | null>(null)

// ============== Computed ==============
const sentimentConfig = computed(() => {
  if (!result.value) return null
  const sentiment = result.value.sentiment
  const configs = {
    positive: {
      label: '正面情感',
      color: '#22c55e',
      bgColor: '#dcfce7',
      icon: '😊',
    },
    negative: {
      label: '负面情感',
      color: '#ef4444',
      bgColor: '#fee2e2',
      icon: '😞',
    },
    neutral: {
      label: '中性情感',
      color: '#6b7280',
      bgColor: '#f3f4f6',
      icon: '😐',
    },
  }
  return configs[sentiment]
})

const confidenceBars = computed(() => {
  if (!result.value) return []
  const { confidence_per_class } = result.value
  return [
    { label: '正向', value: confidence_per_class.positive, color: '#22c55e' },
    { label: '负向', value: confidence_per_class.negative, color: '#ef4444' },
    { label: '中性', value: confidence_per_class.neutral, color: '#6b7280' },
  ]
})

// ============== Methods ==============
async function handleAnalyze() {
  if (!inputText.value.trim()) {
    errorMessage.value = '请输入要分析的文本'
    return
  }

  isLoading.value = true
  errorMessage.value = ''

  try {
    const response = await analyzeText(inputText.value)
    result.value = response
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '分析失败，请检查后端服务'
    result.value = null
  } finally {
    isLoading.value = false
  }
}

function clearResult() {
  inputText.value = ''
  result.value = null
  errorMessage.value = ''
}

function formatPercent(value: number): string {
  return (value * 100).toFixed(1) + '%'
}

// Sample texts for quick testing
const sampleTexts = [
  '今天心情太好了！终于完成了这个项目 😊 #开心 #成就感',
  '这个产品太差了，完全不值这个价！💔 #失望 #再也不买',
  '今天天气不错，去公园散步 🚶 #周末 #放松',
]
</script>

<template>
  <div class="app-container">
    <!-- Header -->
    <header class="app-header">
      <h1 class="app-title">基于深度学习的社交媒体情感分析系统</h1>
      <p class="app-subtitle">Social Media Sentiment Analysis with Deep Learning</p>
    </header>

    <!-- Main Content -->
    <main class="app-main">
      <!-- Input Section -->
      <section class="input-section">
        <div class="input-card">
          <h2 class="section-title">📝 输入文本</h2>
          <textarea
            v-model="inputText"
            class="text-input"
            placeholder="请输入社交媒体文本，支持 Emoji 和话题标签..."
            rows="5"
            :disabled="isLoading"
          ></textarea>

          <!-- Sample texts -->
          <div class="sample-texts">
            <span class="sample-label">示例文本：</span>
            <button
              v-for="(sample, index) in sampleTexts"
              :key="index"
              class="sample-btn"
              @click="inputText = sample"
              :disabled="isLoading"
            >
              {{ sample.slice(0, 15) }}...
            </button>
          </div>

          <!-- Action buttons -->
          <div class="action-buttons">
            <button
              class="analyze-btn"
              @click="handleAnalyze"
              :disabled="isLoading || !inputText.trim()"
            >
              <span v-if="isLoading" class="loading-spinner"></span>
              <span v-else>🔍 开始分析</span>
            </button>
            <button
              class="clear-btn"
              @click="clearResult"
              :disabled="isLoading"
            >
              🗑️ 清空
            </button>
          </div>
        </div>
      </section>

      <!-- Error Message -->
      <section v-if="errorMessage" class="error-section">
        <div class="error-card">
          <span class="error-icon">⚠️</span>
          <span class="error-text">{{ errorMessage }}</span>
        </div>
      </section>

      <!-- Results Section -->
      <section v-if="result && sentimentConfig" class="result-section">
        <!-- Sentiment Conclusion -->
        <div class="result-card sentiment-card" :style="{ backgroundColor: sentimentConfig.bgColor }">
          <div class="sentiment-header">
            <span class="sentiment-icon">{{ sentimentConfig.icon }}</span>
            <span class="sentiment-label" :style="{ color: sentimentConfig.color }">
              {{ sentimentConfig.label }}
            </span>
          </div>
          <div class="sentiment-confidence">
            <span class="confidence-label">置信度</span>
            <span class="confidence-value" :style="{ color: sentimentConfig.color }">
              {{ formatPercent(result.confidence) }}
            </span>
          </div>
          <p class="raw-text">原文：{{ result.raw_text }}</p>
        </div>

        <!-- Confidence Distribution -->
        <div class="result-card">
          <h3 class="card-title">📊 置信度分布</h3>
          <div class="confidence-bars">
            <div
              v-for="bar in confidenceBars"
              :key="bar.label"
              class="confidence-bar-item"
            >
              <div class="bar-header">
                <span class="bar-label">{{ bar.label }}</span>
                <span class="bar-value">{{ formatPercent(bar.value) }}</span>
              </div>
              <div class="bar-track">
                <div
                  class="bar-fill"
                  :style="{ width: formatPercent(bar.value), backgroundColor: bar.color }"
                ></div>
              </div>
            </div>
          </div>
        </div>

        <!-- Social Features -->
        <div class="result-card">
          <h3 class="card-title">🎯 社交特征提取</h3>

          <!-- Cleaned Text -->
          <div class="feature-item">
            <span class="feature-label">清洗后文本：</span>
            <p class="cleaned-text">{{ result.social_features.cleaned_text }}</p>
          </div>

          <!-- Emoji Features -->
          <div class="feature-item" v-if="result.social_features.emoji_descriptions.length > 0">
            <span class="feature-label">
              Emoji 分析 ({{ result.social_features.emoji_count }})：
            </span>
            <div class="tag-list">
              <span
                v-for="emoji in result.social_features.emoji_descriptions"
                :key="emoji"
                class="tag emoji-tag"
              >
                {{ emoji }}
              </span>
            </div>
          </div>

          <!-- Hashtags -->
          <div class="feature-item" v-if="result.social_features.hashtags.length > 0">
            <span class="feature-label">
              话题标签 ({{ result.social_features.hashtag_count }})：
            </span>
            <div class="tag-list">
              <span
                v-for="tag in result.social_features.hashtags"
                :key="tag"
                class="tag hashtag-tag"
              >
                {{ tag }}
              </span>
            </div>
          </div>

          <!-- Mentions -->
          <div class="feature-item" v-if="result.social_features.mentions.length > 0">
            <span class="feature-label">
              @提及 ({{ result.social_features.mention_count }})：
            </span>
            <div class="tag-list">
              <span
                v-for="mention in result.social_features.mentions"
                :key="mention"
                class="tag mention-tag"
              >
                {{ mention }}
              </span>
            </div>
          </div>

          <!-- Statistics -->
          <div class="feature-stats">
            <div class="stat-item">
              <span class="stat-icon">🔗</span>
              <span class="stat-label">URLs 移除</span>
              <span class="stat-value">{{ result.social_features.url_count }}</span>
            </div>
            <div class="stat-item">
              <span class="stat-icon">❗</span>
              <span class="stat-label">感叹号</span>
              <span class="stat-value">{{ result.social_features.exclamation_count }}</span>
            </div>
            <div class="stat-item">
              <span class="stat-icon">❓</span>
              <span class="stat-label">问号</span>
              <span class="stat-value">{{ result.social_features.question_count }}</span>
            </div>
          </div>
        </div>

        <!-- Raw JSON (collapsible) -->
        <details class="result-card json-card">
          <summary class="json-summary">📄 原始 JSON 响应</summary>
          <pre class="json-content">{{ JSON.stringify(result, null, 2) }}</pre>
        </details>
      </section>
    </main>

    <!-- Footer -->
    <footer class="app-footer">
      <p>基于 RoBERTa-wwm-ext + 社交特征融合的深度学习情感分析系统</p>
    </footer>
  </div>
</template>

<style scoped>
/* ============== Layout ============== */
.app-container {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.app-header {
  text-align: center;
  padding: 2rem 1rem;
  color: white;
}

.app-title {
  font-size: 2rem;
  font-weight: 700;
  margin: 0 0 0.5rem 0;
  text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
}

.app-subtitle {
  font-size: 1rem;
  opacity: 0.9;
  margin: 0;
}

.app-main {
  flex: 1;
  max-width: 900px;
  width: 100%;
  margin: 0 auto;
  padding: 0 1rem 2rem 1rem;
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.app-footer {
  text-align: center;
  padding: 1.5rem;
  color: rgba(255, 255, 255, 0.8);
  font-size: 0.875rem;
}

/* ============== Input Section ============== */
.input-section {
  background: white;
  border-radius: 1rem;
  padding: 1.5rem;
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.15);
}

.input-card {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.section-title {
  font-size: 1.25rem;
  font-weight: 600;
  margin: 0;
  color: #1f2937;
}

.text-input {
  width: 100%;
  padding: 1rem;
  border: 2px solid #e5e7eb;
  border-radius: 0.75rem;
  font-size: 1rem;
  font-family: inherit;
  resize: vertical;
  transition: border-color 0.2s, box-shadow 0.2s;
  box-sizing: border-box;
}

.text-input:focus {
  outline: none;
  border-color: #667eea;
  box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
}

.text-input:disabled {
  background-color: #f9fafb;
  cursor: not-allowed;
}

.sample-texts {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  align-items: center;
}

.sample-label {
  font-size: 0.875rem;
  color: #6b7280;
}

.sample-btn {
  padding: 0.375rem 0.75rem;
  font-size: 0.75rem;
  background: #f3f4f6;
  border: 1px solid #e5e7eb;
  border-radius: 9999px;
  cursor: pointer;
  transition: all 0.2s;
  color: #4b5563;
}

.sample-btn:hover:not(:disabled) {
  background: #e5e7eb;
}

.sample-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.action-buttons {
  display: flex;
  gap: 1rem;
}

.analyze-btn {
  flex: 1;
  padding: 0.875rem 1.5rem;
  font-size: 1rem;
  font-weight: 600;
  color: white;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border: none;
  border-radius: 0.75rem;
  cursor: pointer;
  transition: transform 0.2s, box-shadow 0.2s;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
}

.analyze-btn:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
}

.analyze-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
  transform: none;
}

.clear-btn {
  padding: 0.875rem 1.5rem;
  font-size: 1rem;
  font-weight: 500;
  color: #6b7280;
  background: #f3f4f6;
  border: 1px solid #e5e7eb;
  border-radius: 0.75rem;
  cursor: pointer;
  transition: all 0.2s;
}

.clear-btn:hover:not(:disabled) {
  background: #e5e7eb;
  color: #4b5563;
}

.clear-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.loading-spinner {
  width: 1.25rem;
  height: 1.25rem;
  border: 2px solid rgba(255, 255, 255, 0.3);
  border-top-color: white;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

/* ============== Error Section ============== */
.error-section {
  animation: slideIn 0.3s ease;
}

@keyframes slideIn {
  from {
    opacity: 0;
    transform: translateY(-10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.error-card {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 1rem 1.5rem;
  background: #fef2f2;
  border: 1px solid #fecaca;
  border-radius: 0.75rem;
  color: #dc2626;
}

.error-icon {
  font-size: 1.25rem;
}

.error-text {
  font-size: 0.9375rem;
}

/* ============== Result Section ============== */
.result-section {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
  animation: fadeIn 0.5s ease;
}

@keyframes fadeIn {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}

.result-card {
  background: white;
  border-radius: 1rem;
  padding: 1.5rem;
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
}

/* Sentiment Card */
.sentiment-card {
  text-align: center;
  padding: 2rem;
}

.sentiment-header {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 1rem;
  margin-bottom: 1rem;
}

.sentiment-icon {
  font-size: 3rem;
}

.sentiment-label {
  font-size: 2rem;
  font-weight: 700;
}

.sentiment-confidence {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  margin-bottom: 1rem;
}

.confidence-label {
  font-size: 1rem;
  color: #6b7280;
}

.confidence-value {
  font-size: 1.5rem;
  font-weight: 600;
}

.raw-text {
  margin: 0;
  font-size: 0.875rem;
  color: #6b7280;
  font-style: italic;
}

/* Card Title */
.card-title {
  font-size: 1.125rem;
  font-weight: 600;
  margin: 0 0 1.25rem 0;
  color: #1f2937;
  border-bottom: 2px solid #e5e7eb;
  padding-bottom: 0.75rem;
}

/* Confidence Bars */
.confidence-bars {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.confidence-bar-item {
  display: flex;
  flex-direction: column;
  gap: 0.375rem;
}

.bar-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.bar-label {
  font-size: 0.9375rem;
  font-weight: 500;
  color: #374151;
}

.bar-value {
  font-size: 0.875rem;
  font-weight: 600;
  color: #4b5563;
}

.bar-track {
  height: 0.75rem;
  background: #e5e7eb;
  border-radius: 9999px;
  overflow: hidden;
}

.bar-fill {
  height: 100%;
  border-radius: 9999px;
  transition: width 0.5s ease;
}

/* Social Features */
.feature-item {
  margin-bottom: 1.25rem;
}

.feature-item:last-child {
  margin-bottom: 0;
}

.feature-label {
  display: block;
  font-size: 0.875rem;
  font-weight: 600;
  color: #4b5563;
  margin-bottom: 0.5rem;
}

.cleaned-text {
  margin: 0;
  padding: 0.75rem 1rem;
  background: #f9fafb;
  border-radius: 0.5rem;
  font-size: 0.9375rem;
  color: #374151;
  line-height: 1.6;
}

.tag-list {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.tag {
  display: inline-flex;
  align-items: center;
  padding: 0.375rem 0.75rem;
  font-size: 0.8125rem;
  font-weight: 500;
  border-radius: 9999px;
  transition: transform 0.2s;
}

.tag:hover {
  transform: scale(1.05);
}

.emoji-tag {
  background: #fef3c7;
  color: #92400e;
}

.hashtag-tag {
  background: #dbeafe;
  color: #1e40af;
}

.mention-tag {
  background: #e0e7ff;
  color: #3730a3;
}

.feature-stats {
  display: flex;
  gap: 1.5rem;
  margin-top: 1.25rem;
  padding-top: 1.25rem;
  border-top: 1px solid #e5e7eb;
}

.stat-item {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.stat-icon {
  font-size: 1rem;
}

.stat-label {
  font-size: 0.8125rem;
  color: #6b7280;
}

.stat-value {
  font-size: 0.875rem;
  font-weight: 600;
  color: #374151;
}

/* JSON Card */
.json-card {
  background: #1f2937;
  color: #e5e7eb;
}

.json-summary {
  font-size: 0.9375rem;
  font-weight: 500;
  cursor: pointer;
  padding: 0.25rem 0;
}

.json-summary:hover {
  color: white;
}

.json-content {
  margin: 1rem 0 0 0;
  padding: 1rem;
  background: #111827;
  border-radius: 0.5rem;
  font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
  font-size: 0.75rem;
  line-height: 1.6;
  overflow-x: auto;
  white-space: pre-wrap;
  word-break: break-all;
}
</style>
