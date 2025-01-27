use tak::*;

#[test]
fn rotate_even() {
    // corner
    let pos: Pos<6> = Pos { x: 0, y: 0 };
    assert_eq!(pos.rotate(), Pos { x: 0, y: 5 });
    assert_eq!(pos.rotate().rotate(), Pos { x: 5, y: 5 });
    assert_eq!(pos.rotate().rotate().rotate(), Pos { x: 5, y: 0 });
    assert_eq!(pos.rotate().rotate().rotate().rotate(), Pos { x: 0, y: 0 });
    // centre
    let pos: Pos<6> = Pos { x: 2, y: 2 };
    assert_eq!(pos.rotate(), Pos { x: 2, y: 3 });
    assert_eq!(pos.rotate().rotate(), Pos { x: 3, y: 3 });
    assert_eq!(pos.rotate().rotate().rotate(), Pos { x: 3, y: 2 });
    assert_eq!(pos.rotate().rotate().rotate().rotate(), Pos { x: 2, y: 2 });
}

#[test]
fn rotate_odd() {
    // corner
    let pos: Pos<7> = Pos { x: 0, y: 0 };
    assert_eq!(pos.rotate(), Pos { x: 0, y: 6 });
    assert_eq!(pos.rotate().rotate(), Pos { x: 6, y: 6 });
    assert_eq!(pos.rotate().rotate().rotate(), Pos { x: 6, y: 0 });
    assert_eq!(pos.rotate().rotate().rotate().rotate(), Pos { x: 0, y: 0 });
    // centre
    let pos: Pos<7> = Pos { x: 3, y: 3 };
    assert_eq!(pos.rotate(), Pos { x: 3, y: 3 });
    assert_eq!(pos.rotate().rotate(), Pos { x: 3, y: 3 });
    assert_eq!(pos.rotate().rotate().rotate(), Pos { x: 3, y: 3 });
    assert_eq!(pos.rotate().rotate().rotate().rotate(), Pos { x: 3, y: 3 });
}

#[test]
fn mirror_even() {
    let pos: Pos<6> = Pos { x: 1, y: 2 };
    assert_eq!(pos.mirror(), Pos { x: 1, y: 3 });
    assert_eq!(pos.mirror().mirror(), Pos { x: 1, y: 2 });
}

#[test]
fn mirror_odd() {
    let pos: Pos<7> = Pos { x: 4, y: 1 };
    assert_eq!(pos.mirror(), Pos { x: 4, y: 5 });
    assert_eq!(pos.mirror().mirror(), Pos { x: 4, y: 1 });

    // centre line
    let pos: Pos<7> = Pos { x: 2, y: 3 };
    assert_eq!(pos.mirror(), Pos { x: 2, y: 3 });
}

#[test]
fn symmetrical_boards() -> StrResult<()> {
    let [mut g0, mut g1, mut g2, mut g3, mut g4, mut g5, mut g6, mut g7] = Game::<5>::default().symmetries();
    while matches!(g0.winner(), GameResult::Ongoing) {
        let turns = g0.possible_turns();
        let turn = turns.into_iter().next().unwrap();
        println!("{:#?}", turn.clone().symmetries());
        let [t0, t1, t2, t3, t4, t5, t6, t7] = turn.symmetries();
        g0.play(t0)?;
        g1.play(t1)?;
        g2.play(t2)?;
        g3.play(t3)?;
        g4.play(t4)?;
        g5.play(t5)?;
        g6.play(t6)?;
        g7.play(t7)?;
    }
    assert_eq!(g0.winner(), g1.winner());
    assert_eq!(g1.winner(), g2.winner());
    assert_eq!(g2.winner(), g3.winner());
    assert_eq!(g4.winner(), g5.winner());
    assert_eq!(g5.winner(), g6.winner());
    assert_eq!(g6.winner(), g7.winner());
    Ok(())
}
