//! commands/conflict.rs — Расчет матрицы асимметричных конфликтов J на Rust
#[tauri::command]
pub fn compute_conflict_matrix(characters: Vec<String>) -> Vec<Vec<f64>> {
    let n = characters.len();
    let mut j_mat = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            if i != j {
                j_mat[i][j] = (i as f64 - j as f64) * 0.1;
            }
        }
    }
    j_mat
}
