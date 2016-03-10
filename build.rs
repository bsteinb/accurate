extern crate cupid;
extern crate skeptic;

fn main() {
    skeptic::generate_doc_tests(&["README.md"]);

    if let Some(information) = cupid::master() {
        if information.fma() {
            println!("cargo:rustc-cfg=feature=fma");
        }
    }
}
