import os
from collections import defaultdict, Counter
import re


class CustomPorterStemmer:
    """
    Uses Porter Stemming Algorithm.
    Reduces words to their root form, ex running to run.
    """
    
    def __init__(self):
        # Vowels for Porter algorithm
        self.vowels = "aeiou"
        
        # Common suffix patterns and their replacements
        self.step1a_suffixes = [
            ('sses', 'ss'),   # processes -> process
            ('ies', 'i'),     # ties -> ti
            ('ss', 'ss'),     # stress -> stress
            ('s', '')         # cats -> cat
        ]
        
        self.step1b_suffixes = [
            ('eed', 'ee'),    # agreed -> agree
            ('ed', ''),       # played -> play
            ('ing', '')       # playing -> play
        ]
        
        self.step2_suffixes = [
            ('ational', 'ate'),  # relational -> relate
            ('tional', 'tion'),  # conditional -> condition
            ('enci', 'ence'),    # valenci -> valence
            ('anci', 'ance'),    # reluctanci -> reluctance
            ('izer', 'ize'),     # digitizer -> digitize
            ('abli', 'able'),    # conformabli -> conformable
            ('alli', 'al'),      # radicalli -> radical
            ('entli', 'ent'),    # differentli -> different
            ('eli', 'e'),        # vileli -> vile
            ('ousli', 'ous'),    # analogousli -> analogous
            ('ization', 'ize'),  # vietnamization -> vietnamize
            ('ation', 'ate'),    # predication -> predicate
            ('ator', 'ate'),     # operator -> operate
            ('alism', 'al'),     # feudalism -> feudal
            ('iveness', 'ive'),  # decisiveness -> decisive
            ('fulness', 'ful'),  # hopefulness -> hopeful
            ('ousness', 'ous'),  # callousness -> callous
            ('aliti', 'al'),     # formaliti -> formal
            ('iviti', 'ive'),    # sensitiviti -> sensitive
            ('biliti', 'ble')    # sensibiliti -> sensible
        ]
        
        self.step3_suffixes = [
            ('icate', 'ic'),     # triplicate -> triplic
            ('ative', ''),       # formative -> form
            ('alize', 'al'),     # formalize -> formal
            ('iciti', 'ic'),     # electriciti -> electric
            ('ical', 'ic'),      # electrical -> electric
            ('ful', ''),         # hopeful -> hope
            ('ness', '')         # goodness -> good
        ]
        
        self.step4_suffixes = [
            'al', 'ance', 'ence', 'er', 'ic', 'able', 'ible', 'ant', 'ement',
            'ment', 'ent', 'ion', 'ou', 'ism', 'ate', 'iti', 'ous', 'ive', 'ize'
        ]
        
        self.step5_suffixes = [
            ('e', ''),           # rate -> rat
            ('ll', 'l')          # roll -> rol
        ]

    def _is_vowel(self, word, i):
        """Check if character at position i is a vowel in context."""
        if i < 0 or i >= len(word):
            return False
        
        char = word[i].lower()
        if char in self.vowels:
            return True
        
        # 'y' is a vowel if preceded by a consonant
        if char == 'y' and i > 0:
            return not self._is_vowel(word, i-1)
        
        return False

    def _measure(self, word):
        """
        Calculate the measure of a word (number of vowel-consonant sequences).
        This is crucial for Porter algorithm rules.
        """
        n = len(word)
        if n == 0:
            return 0
        
        # Find vowel-consonant pattern
        i = 0
        # Skip initial consonants
        while i < n and not self._is_vowel(word, i):
            i += 1
        
        measure = 0
        while i < n:
            # Skip vowels
            while i < n and self._is_vowel(word, i):
                i += 1
            if i >= n:
                break
            
            # Skip consonants
            while i < n and not self._is_vowel(word, i):
                i += 1
            measure += 1
        
        return measure

    def _contains_vowel(self, word):
        """Check if word contains any vowel."""
        return any(self._is_vowel(word, i) for i in range(len(word)))

    def _ends_with_double_consonant(self, word):
        """Check if word ends with double consonant."""
        if len(word) < 2:
            return False
        return (word[-1] == word[-2] and 
                not self._is_vowel(word, len(word)-1) and
                not self._is_vowel(word, len(word)-2))

    def _cvc_pattern(self, word):
        """Check if word ends with consonant-vowel-consonant pattern."""
        if len(word) < 3:
            return False
        
        i = len(word) - 3
        return (not self._is_vowel(word, i) and 
                self._is_vowel(word, i+1) and 
                not self._is_vowel(word, i+2) and
                word[i+2] not in 'wxy')

    def stem(self, word):
        """Apply Porter Stemming Algorithm to reduce word to its stem."""
        if len(word) <= 2:
            return word.lower()
        
        word = word.lower().strip()
        original_word = word
        
        # Step 1a: Handle plurals
        for suffix, replacement in self.step1a_suffixes:
            if word.endswith(suffix):
                word = word[:-len(suffix)] + replacement
                break
        
        # Step 1b: Handle past tense
        for suffix, replacement in self.step1b_suffixes:
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if suffix == 'eed':
                    if self._measure(stem) > 0:
                        word = stem + replacement
                else:  # 'ed' or 'ing'
                    if self._contains_vowel(stem):
                        word = stem
                        # Post-processing for step 1b
                        if word.endswith(('at', 'bl', 'iz')):
                            word += 'e'
                        elif self._ends_with_double_consonant(word) and not word.endswith(('l', 's', 'z')):
                            word = word[:-1]
                        elif self._measure(word) == 1 and self._cvc_pattern(word):
                            word += 'e'
                break
        
        # Step 2: Handle other suffixes
        for suffix, replacement in self.step2_suffixes:
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if self._measure(stem) > 0:
                    word = stem + replacement
                break
        
        # Step 3: Handle more suffixes
        for suffix, replacement in self.step3_suffixes:
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if self._measure(stem) > 0:
                    word = stem + replacement
                break
        
        # Step 4: Remove suffixes
        for suffix in self.step4_suffixes:
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if self._measure(stem) > 1:
                    word = stem
                    break
                elif suffix == 'ion' and self._measure(stem) > 1 and stem.endswith(('s', 't')):
                    word = stem
                    break
        
        # Step 5: Final cleanup
        for suffix, replacement in self.step5_suffixes:
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if suffix == 'e':
                    if self._measure(stem) > 1 or (self._measure(stem) == 1 and not self._cvc_pattern(stem)):
                        word = stem
                elif suffix == 'll':
                    if self._measure(stem) > 1:
                        word = stem + 'l'
                break
        
        return word


class SearchEngine:
    def __init__(self, filepath):
        """
        Initializes the search engine by loading and indexing data.
        """
        print("\nInitializing custom search engine...")
        
        # Initialize custom components
        self.stemmer = CustomPorterStemmer()
        self.stop_words = self._get_stopwords()
        
        self.indexed_data = self._load_indexed_data(filepath)
        if self.indexed_data:
            self._build_index()
            print("Initialization complete.")
            # Add debug information
            print(f"Debug: Index contains {len(self.inverted_index)} unique terms")
            
            # Show some sample terms for debugging
            sample_terms = list(self.inverted_index.keys())[:10]
            print(f"Debug: Sample indexed terms: {sample_terms}")
        else:
            print("Initialization failed: No data loaded.")

    def _get_stopwords(self):
        """
        Custom stopwords list - common English words that don't add search value.
        """
        return set([
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
            'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
            'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
            'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after',
            'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
            'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
            'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren',
            'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn',
            'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn'
        ])

    def _custom_tokenize(self, text):
        """
        FIXED: Custom tokenizer that splits text into words.
        Now handles URLs, product names, and mixed alphanumeric content better.
        """
        # Convert to lowercase
        text = text.lower()
        
        # IMPROVED: Instead of removing all non-alphanumeric, preserve important patterns
        # Split by whitespace first, then clean each token
        raw_tokens = text.split()
        tokens = []
        
        for token in raw_tokens:
            # Remove surrounding punctuation but keep internal structure
            # This preserves things like "ipad", "iphone", "macbook-pro" etc.
            cleaned = re.sub(r'^[^\w]+|[^\w]+$', '', token)
            
            # Split on remaining punctuation but keep the parts
            parts = re.split(r'[^\w]+', cleaned)
            
            for part in parts:
                if part and len(part) > 0:  # FIXED: Changed condition
                    tokens.append(part)
        
        return tokens

    def _load_indexed_data(self, filepath):
        """
        IMPROVED: Loads and parses the crawled data from a file
        """
        indexed_data = []
        if not os.path.exists(filepath):
            print(f"Error: File not found at '{filepath}'")
            return indexed_data
            
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                
                # IMPROVED: Better parsing logic
                entries = content.split("-" * 80)
                
                for doc_id, entry in enumerate(entries):
                    entry = entry.strip()
                    if entry:
                        lines = entry.split("\n")
                        
                        # Find the URL line (should start with "URL:")
                        url = ""
                        text_start_idx = 0
                        
                        for i, line in enumerate(lines):
                            if line.startswith("URL:"):
                                url = line.replace("URL:", "").strip()
                                text_start_idx = i + 1
                                break
                        
                        # Everything after URL line is content
                        if text_start_idx < len(lines):
                            text = "\n".join(lines[text_start_idx:]).strip()
                            if text:  # Only add if there's actual content
                                indexed_data.append({
                                    'id': doc_id, 
                                    'url': url, 
                                    'text': text
                                })
                                
        except Exception as e:
            print(f"Error reading file: {e}")
            
        print(f"Debug: Loaded {len(indexed_data)} documents from file")
        return indexed_data

    def _preprocess_text(self, text):
        """
        IMPROVED: Processes text using custom tokenizer, stopword removal, and stemmer.
        """
        # Custom tokenization
        tokens = self._custom_tokenize(text)
        
        # IMPROVED: More flexible filtering
        processed_tokens = []
        for token in tokens:
            # FIXED: Less restrictive filtering - keep tokens that are mostly alphabetic
            if (len(token) > 0 and  # Keep any non-empty token
                not token.isdigit() and  # Skip pure numbers
                token not in self.stop_words):  # Skip stopwords
                
                # Apply stemming only to purely alphabetic words
                if token.isalpha():
                    stemmed = self.stemmer.stem(token)
                    processed_tokens.append(stemmed)
                else:
                    # Keep mixed alphanumeric terms as-is (like "ipad", "iphone")
                    processed_tokens.append(token)
        
        return processed_tokens

    def _build_index(self):
        """
        Builds the inverted index and calculates document statistics needed for BM25.
        """
        self.inverted_index = defaultdict(list)
        self.doc_lengths = {}
        
        for doc in self.indexed_data:
            doc_id = doc['id']
            # IMPROVED: Process both URL and text content for better matching
            full_content = doc['url'] + " " + doc['text']
            processed_words = self._preprocess_text(full_content)
            
            # Debug: Print processed words for first few documents
            if doc_id < 3:
                print(f"Debug Doc {doc_id}: {doc['url']}")
                print(f"Debug Processed words sample: {processed_words[:10]}")
            
            # Store document length (number of terms)
            self.doc_lengths[doc_id] = len(processed_words)
            
            # Count word frequencies in this document
            word_counts = Counter(processed_words)
            
            # Build inverted index: word -> list of (doc_id, term_frequency)
            for word, freq in word_counts.items():
                self.inverted_index[word].append((doc_id, freq))
        
        # Calculate average document length for BM25
        if self.doc_lengths:
            self.avg_doc_length = sum(self.doc_lengths.values()) / len(self.doc_lengths)
        else:
            self.avg_doc_length = 0

    def _natural_log(self, x):
        """
        Custom implementation of natural logarithm using Taylor series.
        ln(x) ≈ 2 * ((x-1)/(x+1) + (1/3)*((x-1)/(x+1))^3 + (1/5)*((x-1)/(x+1))^5 + ...)
        """
        if x <= 0:
            return float('-inf')
        if x == 1:
            return 0
        
        # For better convergence, use the identity: ln(x) = ln(x/e^k) + k
        # We'll normalize x to be close to 1
        k = 0
        e_approx = 2.718281828459045
        
        while x > e_approx:
            x /= e_approx
            k += 1
        while x < 1/e_approx:
            x *= e_approx
            k -= 1
        
        # Now use Taylor series for ln(x) where x is close to 1
        # ln(x) = ln(1 + (x-1)) = (x-1) - (x-1)^2/2 + (x-1)^3/3 - (x-1)^4/4 + ...
        z = x - 1
        result = 0
        term = z
        
        for i in range(1, 50):  # Use 50 terms for good accuracy
            if i % 2 == 1:
                result += term / i
            else:
                result -= term / i
            term *= z
            
            # Stop if term becomes very small
            if abs(term / i) < 1e-10:
                break
        
        return result + k

    def _calculate_bm25_score(self, query_words, k1=1.5, b=0.75):
        """
        FIXED: Custom implementation of BM25 scoring algorithm.
        
        BM25 Formula:
        Score(D,Q) = Σ IDF(qi) * (f(qi,D) * (k1 + 1)) / (f(qi,D) + k1 * (1 - b + b * (|D| / avgdl)))
        
        Where:
        - IDF(qi) = ln((N - df(qi) + 0.5) / (df(qi) + 0.5))
        - f(qi,D) = frequency of term qi in document D
        - |D| = length of document D
        - avgdl = average document length
        - N = total number of documents
        - df(qi) = number of documents containing qi
        - k1, b = tuning parameters
        """
        scores = defaultdict(float)
        N = len(self.indexed_data)  # Total number of documents
        
        for word in query_words:
            if word in self.inverted_index:
                postings = self.inverted_index[word]
                df = len(postings)  # Document frequency: how many docs contain this word
                
                print(f"Debug: Word '{word}' appears in {df} out of {N} documents")
                
                # Calculate IDF (Inverse Document Frequency)
                # IDF = ln((N - df + 0.5) / (df + 0.5))
                numerator = N - df + 0.5
                denominator = df + 0.5
                
                print(f"Debug: IDF calculation: ln({numerator}/{denominator})")
                
                # FIXED: Handle cases where df is very high (negative IDF)
                if numerator <= 0:
                    # When a term appears in most documents, use a small positive IDF
                    idf = 0.01
                    print(f"Debug: Very common term, using small IDF: {idf}")
                else:
                    idf = self._natural_log(numerator / denominator)
                    print(f"Debug: Calculated IDF: {idf}")
                
                # FIXED: If IDF is negative or very small for common terms, 
                # use a minimum positive value to ensure scoring works
                if idf <= 0:
                    idf = 0.01
                    print(f"Debug: Adjusted negative IDF to: {idf}")
                
                # Calculate BM25 score for each document containing this word
                for doc_id, term_freq in postings:
                    doc_length = self.doc_lengths.get(doc_id, 1)  # FIXED: Handle missing doc_id
                    
                    # Avoid division by zero
                    if self.avg_doc_length == 0:
                        continue
                    
                    # BM25 term frequency component
                    # (f(qi,D) * (k1 + 1)) / (f(qi,D) + k1 * (1 - b + b * (|D| / avgdl)))
                    tf_numerator = term_freq * (k1 + 1)
                    tf_denominator = term_freq + k1 * (1 - b + b * (doc_length / self.avg_doc_length))
                    
                    tf_component = tf_numerator / tf_denominator
                    
                    # Final BM25 score for this term in this document
                    term_score = idf * tf_component
                    scores[doc_id] += term_score
                    
                    # Debug first few scores
                    if len(scores) <= 3:
                        print(f"Debug: Doc {doc_id} - tf:{term_freq}, tf_comp:{tf_component:.4f}, term_score:{term_score:.4f}")
        
        # Debug final scores
        if scores:
            max_score = max(scores.values())
            min_score = min(scores.values())
            print(f"Debug: Score range: {min_score:.4f} to {max_score:.4f}")
        
        return scores

    def search(self, query, max_results=5, k1=1.5, b=0.75):
        """
        IMPROVED: Searches the index using custom BM25 implementation with debugging.
        
        Parameters:
        - k1: Controls term frequency saturation (1.2-2.0 typical)
        - b: Controls length normalization (0.75 typical)
        """
        print(f"Debug: Searching for '{query}'")
        
        query_words = self._preprocess_text(query)
        print(f"Debug: Processed query words: {query_words}")
        
        if not query_words:
            print("Debug: No valid query words after preprocessing")
            return []
        
        # Check if query terms exist in index
        for word in query_words:
            if word in self.inverted_index:
                print(f"Debug: Found '{word}' in {len(self.inverted_index[word])} documents")
            else:
                print(f"Debug: '{word}' not found in index")
        
        # Calculate BM25 scores for all documents
        scores = self._calculate_bm25_score(query_words, k1, b)
        print(f"Debug: Found scores for {len(scores)} documents")
        
        # Sort documents by score (highest first)
        ranked_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top results with document info
        results = []
        for doc_id, score in ranked_results[:max_results]:
            # FIXED: Remove the score > 0 filter that was blocking results
            doc = self.indexed_data[doc_id]
            results.append((doc['url'], doc['text'][:300], score))
            print(f"Debug: Adding result - Doc {doc_id}, Score: {score:.4f}")
        
        return results


# === MAIN ===
if __name__ == "__main__":
    filepath = "extractedText/website_index.txt"
    engine = SearchEngine(filepath)

    if engine.indexed_data:
    
        while True:
            query = input("\nSearch Query (or type 'exit'): ")
            if query.lower() == 'exit':
                break
                
            results = engine.search(query)
            if results:
                print(f"\n--- Top {len(results)} results for '{query}' ---")
                for i, (url, snippet, score) in enumerate(results, 1):
                    print(f"\nResult {i}: {url}")
                    print(f"BM25 Score: {score:.4f}")
                    print(f"Preview: {snippet}...")
                    print("-" * 60)
            else:
                print("No results found.")
    else:
        print("Failed to load data. Please check your file path.")