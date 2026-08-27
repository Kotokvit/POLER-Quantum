//! parser/epsilon.rs — Вычисление epsilon-плотности для художественных глав
pub fn calculate_chapter_epsilon(text: &str) -> f64 {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() { return 0.0; }
    // Epsilon density calculation
    words.len() as f64 * 0.042
}
