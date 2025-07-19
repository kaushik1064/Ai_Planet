// Utility functions for Math AI Assistant
const MathUtils = {
    /**
     * Extract mathematical expressions from text
     * @param {string} text - Input text to analyze
     * @returns {string[]} Array of mathematical expressions found
     */
    extractMathematicalExpressions(text) {
        if (!text) return [];
        
        // Patterns for mathematical expressions
        const patterns = [
            /\$[^$]+\$/g,  // LaTeX inline math
            /\\\([^)]+\\\)/g,  // LaTeX inline math alternative
            /\\\[[^\]]+\\\]/g,  // LaTeX display math
            /\\begin\{[^}]+\}.*?\\end\{[^}]+\}/g,  // LaTeX environments
            /[0-9]+\s*[+\-*/=<>≤≥≠]\s*[0-9]+/g,  // Simple arithmetic
            /[a-zA-Z]\s*[=]\s*[^,\s]+/g,  // Variable assignments
            /∫[^∫]*d[a-zA-Z]/g,  // Integrals
            /∑[^∑]*/g,  // Summations
            /√[^√\s]*/g  // Square roots
        ];
        
        const expressions = new Set();  // Using Set to avoid duplicates
        
        patterns.forEach(pattern => {
            const matches = text.match(pattern) || [];
            matches.forEach(match => expressions.add(match));
        });
        
        return Array.from(expressions);
    },

    /**
     * Clean and normalize mathematical text
     * @param {string} text - Input text to clean
     * @returns {string} Cleaned text
     */
    cleanMathematicalText(text) {
        if (!text) return "";
        
        // Remove excessive whitespace
        let cleaned = text.replace(/\s+/g, ' ');
        
        // Normalize mathematical symbols
        const replacements = {
            '×': '*',
            '÷': '/',
            '–': '-',
            '—': '-',
            '≤': '<=',
            '≥': '>=',
            '≠': '!=',
            '→': '->',
            '∞': 'infinity'
        };
        
        Object.entries(replacements).forEach(([oldChar, newChar]) => {
            cleaned = cleaned.split(oldChar).join(newChar);
        });
        
        // Clean up spacing around operators
        cleaned = cleaned.replace(/\s*([+\-*/=<>])\s*/g, ' $1 ');
        
        return cleaned.trim();
    },

    /**
     * Extract numerical answer from solution text
     * @param {string} text - Solution text to analyze
     * @returns {string|null} Extracted numerical answer or null if not found
     */
    extractNumericalAnswer(text) {
        if (!text) return null;
        
        // Look for common answer patterns
        const answerPatterns = [
            /(?:answer|result|solution)(?:\s*[:=]?\s*)([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)/i,
            /\\boxed\{([^}]+)\}/,  // LaTeX boxed answers
            /(?:therefore|thus|hence)(?:.*?)([+-]?\d+(?:\.\d+)?)/i,
            /([+-]?\d+(?:\.\d+)?)\s*(?:is\s+the\s+answer|is\s+the\s+result)/i,
            /=\s*([+-]?\d+(?:\.\d+)?)(?:\s|$)/  // Final equals
        ];
        
        for (const pattern of answerPatterns) {
            const matches = text.match(pattern);
            if (matches && matches[1]) {
                return matches[1];
            }
        }
        
        return null;
    },

    /**
     * Validate if a string is a valid mathematical expression
     * @param {string} expression - Expression to validate
     * @returns {boolean} True if valid, false otherwise
     */
    validateMathematicalExpression(expression) {
        if (!expression) return false;
        
        // Check balanced parentheses
        const openCount = (expression.match(/\(/g) || []).length;
        const closeCount = (expression.match(/\)/g) || []).length;
        if (openCount !== closeCount) return false;
        
        // Check for valid mathematical characters
        const validChars = new Set('0123456789+-*/()=<>. abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ√∫∑∏π∞');
        for (const c of expression) {
            if (!validChars.has(c)) return false;
        }
        
        return true;
    }
};

const TextProcessor = {
    /**
     * Truncate text to specified length
     * @param {string} text - Input text
     * @param {number} maxLength - Maximum length
     * @param {string} suffix - Suffix to add if truncated
     * @returns {string} Truncated text
     */
    truncateText(text, maxLength = 100, suffix = "...") {
        if (!text) return "";
        if (text.length <= maxLength) return text;
        return text.substring(0, maxLength - suffix.length) + suffix;
    },

    /**
     * Extract keywords from text
     * @param {string} text - Input text
     * @param {number} minLength - Minimum keyword length
     * @returns {string[]} Array of keywords
     */
    extractKeywords(text, minLength = 3) {
        if (!text) return [];
        
        // Remove punctuation and convert to lowercase
        const cleanText = text.replace(/[^\w\s]/g, ' ').toLowerCase();
        
        // Split into words
        const words = cleanText.split(/\s+/);
        
        // Filter out short words and common stop words
        const stopWords = new Set([
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
        ]);
        
        const keywords = words.filter(
            word => word.length >= minLength && !stopWords.has(word)
        );
        
        // Remove duplicates
        return [...new Set(keywords)];
    },

    /**
     * Calculate simple text similarity using Jaccard index
     * @param {string} text1 - First text
     * @param {string} text2 - Second text
     * @returns {number} Similarity score (0-1)
     */
    calculateTextSimilarity(text1, text2) {
        if (!text1 || !text2) return 0;
        
        // Convert to sets of words
        const words1 = new Set(text1.toLowerCase().split(/\s+/));
        const words2 = new Set(text2.toLowerCase().split(/\s+/));
        
        // Calculate intersection and union
        const intersection = new Set([...words1].filter(word => words2.has(word)));
        const union = new Set([...words1, ...words2]);
        
        if (union.size === 0) return 0;
        return intersection.size / union.size;
    },

    /**
     * Format solution text into structured steps
     * @param {string} solution - Solution text
     * @returns {Array<{step_number: string, description: string}>} Array of steps
     */
    formatSolutionSteps(solution) {
        if (!solution) return [];
        
        const steps = [];
        
        // Split by common step indicators
        const stepPatterns = [
            /(?:Step\s*\d+|First|Second|Third|Fourth|Fifth|Next|Then|Finally)[:.]?\s*/i,
            /\d+\.\s*/,
            /\n\n+/
        ];
        
        // Try each pattern
        for (const pattern of stepPatterns) {
            const parts = solution.split(pattern);
            if (parts.length > 1) {
                for (let i = 1; i < parts.length; i++) {
                    const part = parts[i].trim();
                    if (part) {
                        steps.push({
                            step_number: String(i),
                            description: part
                        });
                    }
                }
                break;
            }
        }
        
        // If no clear steps found, treat as single step
        if (steps.length === 0 && solution.trim()) {
            steps.push({
                step_number: "1",
                description: solution.trim()
            });
        }
        
        return steps;
    },

    /**
     * Normalize text by removing extra spaces and standardizing case
     * @param {string} text - Input text
     * @returns {string} Normalized text
     */
    normalizeText(text) {
        if (!text) return "";
        return text.trim().replace(/\s+/g, ' ').toLowerCase();
    },

    /**
     * Generate a simple summary of text
     * @param {string} text - Input text
     * @param {number} maxLength - Maximum length of summary
     * @returns {string} Summary text
     */
    summarize(text, maxLength = 300) {
        if (!text) return "";
        if (text.length <= maxLength) return text;
        
        // Try to find a good sentence break
        const lastPeriod = text.lastIndexOf('.', maxLength);
        const lastSpace = text.lastIndexOf(' ', maxLength);
        
        const cutoff = lastPeriod > maxLength * 0.7 ? lastPeriod + 1 : 
                      lastSpace > maxLength * 0.7 ? lastSpace : 
                      maxLength;
        
        return text.substring(0, cutoff) + "...";
    }
};

const DataValidator = {
    /**
     * Validate mathematical question format
     * @param {string} question - Question to validate
     * @returns {Object} Validation result
     */
    validateQuestionFormat(question) {
        const result = {
            isValid: true,
            errors: [],
            warnings: []
        };
        
        if (!question || !question.trim()) {
            result.isValid = false;
            result.errors.push("Question cannot be empty");
            return result;
        }
        
        // Check length
        if (question.length > 2000) {
            result.warnings.push("Question is very long");
        } else if (question.length < 5) {
            result.warnings.push("Question is very short");
        }
        
        // Check for mathematical content
        const mathIndicators = [
            'solve', 'find', 'calculate', 'compute', 'determine', 'evaluate',
            '=', '+', '-', '*', '/', '^', '√', '∫', '∑', 'x', 'y', 'equation'
        ];
        
        const lowerQuestion = question.toLowerCase();
        const hasMathContent = mathIndicators.some(indicator => 
            lowerQuestion.includes(indicator));
        
        if (!hasMathContent) {
            result.warnings.push("Question may not be mathematical in nature");
        }
        
        return result;
    },

    /**
     * Validate difficulty level
     * @param {string} difficulty - Difficulty level to validate
     * @returns {boolean} True if valid, false otherwise
     */
    validateDifficultyLevel(difficulty) {
        const validLevels = ["easy", "medium", "hard"];
        return validLevels.includes(difficulty.toLowerCase());
    },

    /**
     * Validate feedback data structure
     * @param {Object} feedback - Feedback data to validate
     * @returns {Object} Validation result
     */
    validateFeedbackData(feedback) {
        const result = {
            isValid: true,
            errors: []
        };
        
        const requiredFields = ["rating"];
        for (const field of requiredFields) {
            if (!(field in feedback)) {
                result.isValid = false;
                result.errors.push(`Missing required field: ${field}`);
            }
        }
        
        // Validate rating
        if ("rating" in feedback) {
            const rating = feedback.rating;
            if (typeof rating !== 'number' || rating < 1 || rating > 5) {
                result.isValid = false;
                result.errors.push("Rating must be an integer between 1 and 5");
            }
        }
        
        return result;
    }
};

// Simple in-memory cache implementation
class CacheUtils {
    constructor(maxSize = 1000, ttlSeconds = 3600) {
        this.cache = new Map();
        this.timestamps = new Map();
        this.maxSize = maxSize;
        this.ttlSeconds = ttlSeconds;
    }

    /**
     * Get value from cache
     * @param {string} key - Cache key
     * @returns {any|null} Cached value or null if not found/expired
     */
    get(key) {
        if (!this.cache.has(key)) return null;
        
        // Check if expired
        if (this._isExpired(key)) {
            this._remove(key);
            return null;
        }
        
        return this.cache.get(key);
    }

    /**
     * Set value in cache
     * @param {string} key - Cache key
     * @param {any} value - Value to cache
     */
    set(key, value) {
        // Remove oldest items if cache is full
        if (this.cache.size >= this.maxSize) {
            this._evictOldest();
        }
        
        this.cache.set(key, value);
        this.timestamps.set(key, new Date());
    }

    /**
     * Remove value from cache
     * @param {string} key - Cache key to remove
     */
    remove(key) {
        this._remove(key);
    }

    /**
     * Clear all cache
     */
    clear() {
        this.cache.clear();
        this.timestamps.clear();
    }

    /**
     * Generate hash for cache key
     * @param {string} text - Text to hash
     * @returns {string} SHA-256 hash
     */
    static generateHash(text) {
        // Simple hash function for demo purposes
        // In production, use a proper cryptographic hash
        let hash = 0;
        for (let i = 0; i < text.length; i++) {
            const char = text.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash; // Convert to 32bit integer
        }
        return hash.toString();
    }

    // Private methods
    _isExpired(key) {
        if (!this.timestamps.has(key)) return true;
        
        const age = (new Date() - this.timestamps.get(key)) / 1000;
        return age > this.ttlSeconds;
    }

    _remove(key) {
        this.cache.delete(key);
        this.timestamps.delete(key);
    }

    _evictOldest() {
        if (this.timestamps.size === 0) return;
        
        let oldestKey = null;
        let oldestTime = Infinity;
        
        for (const [key, time] of this.timestamps) {
            if (time < oldestTime) {
                oldestTime = time;
                oldestKey = key;
            }
        }
        
        if (oldestKey) this._remove(oldestKey);
    }
}

// Simple rate limiter implementation
class RateLimiter {
    constructor(maxRequests = 100, windowSeconds = 60) {
        this.maxRequests = maxRequests;
        this.windowSeconds = windowSeconds;
        this.requests = new Map();
    }

    /**
     * Check if request is allowed for identifier
     * @param {string} identifier - Client identifier
     * @returns {boolean} True if allowed, false if rate limited
     */
    isAllowed(identifier) {
        const now = new Date();
        const windowStart = new Date(now.getTime() - this.windowSeconds * 1000);
        
        // Clean old requests
        if (this.requests.has(identifier)) {
            const timestamps = this.requests.get(identifier).filter(
                time => time > windowStart
            );
            this.requests.set(identifier, timestamps);
        } else {
            this.requests.set(identifier, []);
        }
        
        // Check if under limit
        const currentRequests = this.requests.get(identifier).length;
        if (currentRequests < this.maxRequests) {
            this.requests.get(identifier).push(now);
            return true;
        }
        
        return false;
    }

    /**
     * Get remaining requests for identifier
     * @param {string} identifier - Client identifier
     * @returns {number} Number of remaining requests in window
     */
    getRemainingRequests(identifier) {
        if (!this.requests.has(identifier)) return this.maxRequests;
        
        const now = new Date();
        const windowStart = new Date(now.getTime() - this.windowSeconds * 1000);
        
        const currentRequests = this.requests.get(identifier).filter(
            time => time > windowStart
        ).length;
        
        return Math.max(0, this.maxRequests - currentRequests);
    }
}

// Export all utilities
module.exports = {
    MathUtils,
    TextProcessor,
    DataValidator,
    CacheUtils,
    RateLimiter
};