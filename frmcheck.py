from hypothesis import given, strategies as st
@given(s = st.sampled_from(['1', '2', '3']))
def test(s):
    print(s)
    print()
